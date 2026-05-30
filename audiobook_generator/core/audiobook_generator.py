import logging
import multiprocessing
import os
from collections.abc import Iterable, Iterator

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.utils.log_handler import setup_logging
from audiobook_generator.utils.filename_sanitizer import make_safe_filename

logger = logging.getLogger(__name__)

CHAPTER_SUMMARY_PROMPT = (
    "Your task is to write a concise, FACTUAL and comprehensive summary of the provided text that captures all key events and information, using ONLY the information provided. "
    "The summary must be written entirely in full and well-structured paragraphs. Use a very fucking generous amount of swear words to spice things up. "
    "Do NOT include any introduction, commentary, or acknowledgment of these instructions. Now, get on with the fucking summary."
)


def _make_summary_client(config):
    import openai
    kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "no-key")}
    if config.summary_base_url:
        kwargs["base_url"] = config.summary_base_url
    return openai.OpenAI(**kwargs)


def _invoke_summary(client, model: str, text: str) -> str:
    return (
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CHAPTER_SUMMARY_PROMPT},
                {"role": "user", "content": text.strip()},
            ],
        )
        .choices[0]
        .message.content.strip()
    )


def _split_for_summary(text: str, model: str) -> list:
    import tiktoken
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    max_tokens = 16384 - 2000
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )

    chunks = []
    for chunk in splitter.split_text(text):
        tokens = enc.encode(chunk)
        if len(tokens) <= max_tokens:
            chunks.append(chunk)
        else:
            avg_chars = max(1, len(chunk) // len(tokens))
            safe_size = max_tokens * avg_chars * 8 // 10
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=safe_size,
                chunk_overlap=int(safe_size * 0.1),
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator="end",
            )
            for sub in sub_splitter.split_text(chunk):
                sub_tokens = enc.encode(sub)
                if len(sub_tokens) > max_tokens:
                    sub = enc.decode(sub_tokens[:max_tokens])
                chunks.append(sub)
    return chunks


def _summarize_chapter(text: str, client, model: str) -> str:
    chunks = _split_for_summary(text, model)
    if len(chunks) == 1:
        return _invoke_summary(client, model, chunks[0])
    summaries = [_invoke_summary(client, model, c) for c in chunks]
    return _invoke_summary(client, model, "\n\n".join(summaries))


def chapter_summary_iterator(
    chapters: Iterable,
    client,
    model: str,
) -> Iterator:
    """Yield each chapter followed by an AI-generated summary chapter."""
    for title, text in chapters:
        yield title, text
        summary = None
        for attempt in range(1, 4):
            try:
                result = _summarize_chapter(text, client, model)
                if result.strip():
                    summary = result
                    logger.info(f"Summary for '{title}':\n{summary}")
                    break
                logger.warning(f"Empty summary for '{title}' (attempt {attempt}/3), retrying...")
            except Exception as e:
                logger.warning(f"Summary failed for '{title}' (attempt {attempt}/3): {e}")
        if not summary:
            summary = "Couldn't generate summary after 3 attempts."
        yield f"Summary_of_{title}", f"Chapter summary\n\n{summary}"


def confirm_conversion():
    logger.info("Do you want to continue? (y/n)")
    answer = input()
    if answer.lower() != "y":
        logger.info("Aborted.")
        exit(0)


def get_total_chars(chapters):
    total_characters = 0
    for title, text in chapters:
        total_characters += len(text)
    return total_characters


class AudiobookGenerator:
    def __init__(self, config: GeneralConfig):
        self.config = config

    def __str__(self) -> str:
        return f"{self.config}"

    def process_chapter(self, idx, title, text, book_parser):
        """Process a single chapter: write text (if needed) and convert to audio."""
        try:
            logger.info(f"Processing chapter {idx}: {title}")
            tts_provider = get_tts_provider(self.config)

            # Save chapter text if required
            if self.config.output_text:
                safe_txt_name = make_safe_filename(
                    title=title,
                    idx=idx,
                    output_dir=self.config.output_folder,
                    ext=".txt",
                    collision_check=False,
                )
                text_file = os.path.join(self.config.output_folder, safe_txt_name)
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)

            # Skip audio generation in preview mode
            if self.config.preview:
                return True

            # Generate audio file (safe, length-limited, cross-platform)
            audio_ext = "." + tts_provider.get_output_file_extension()
            safe_audio_name = make_safe_filename(
                title=title,
                idx=idx,
                output_dir=self.config.output_folder,
                ext=audio_ext,
                collision_check=False,
            )
            output_file = os.path.join(self.config.output_folder, safe_audio_name)

            audio_tags = AudioTags(
                title, book_parser.get_book_author(), book_parser.get_book_title(), idx
            )
            tts_provider.text_to_speech(text, output_file, audio_tags)

            logger.info(f"✅ Converted chapter {idx}: {title}, output file: {output_file}")

            return True
        except Exception as e:
            logger.exception(f"Error processing chapter {idx}, error: {e}")
            return False

    def process_chapter_wrapper(self, args):
        """Wrapper for process_chapter to handle unpacking args for imap."""
        idx, title, text, book_parser = args
        return idx, self.process_chapter(idx, title, text, book_parser)

    def run(self):
        try:
            logger.info("Starting audiobook generation...")
            book_parser = get_book_parser(self.config)
            tts_provider = get_tts_provider(self.config)

            os.makedirs(self.config.output_folder, exist_ok=True)

            if self.config.save_cover:
                get_cover = getattr(book_parser, "get_cover", None)
                if get_cover:
                    cover = get_cover()
                    if cover:
                        ext = os.path.splitext(cover.file_name)[1]
                        cover_path = os.path.join(self.config.output_folder, f"cover{ext}")
                        with open(cover_path, "wb") as f:
                            f.write(cover.get_content())
                        logger.info(f"Cover image saved as cover{ext}")

            chapters = book_parser.get_chapters(tts_provider.get_break_string())
            # Filter out empty or very short chapters
            chapters = [(title, text) for title, text in chapters if text.strip()]

            logger.info(f"Chapters count: {len(chapters)}.")

            # Check chapter start and end args
            if self.config.chapter_start < 1 or self.config.chapter_start > len(chapters):
                raise ValueError(
                    f"Chapter start index {self.config.chapter_start} is out of range. Check your input."
                )
            if self.config.chapter_end < -1 or self.config.chapter_end > len(chapters):
                raise ValueError(
                    f"Chapter end index {self.config.chapter_end} is out of range. Check your input."
                )
            if self.config.chapter_end == -1:
                self.config.chapter_end = len(chapters)
            if self.config.chapter_start > self.config.chapter_end:
                raise ValueError(
                    f"Chapter start index {self.config.chapter_start} is larger than chapter end index {self.config.chapter_end}. Check your input."
                )

            logger.info(
                f"Converting chapters from {self.config.chapter_start} to {self.config.chapter_end}."
            )

            # Initialize total_characters to 0
            total_characters = get_total_chars(
                chapters[self.config.chapter_start - 1 : self.config.chapter_end]
            )
            logger.info(f"Total characters in selected book chapters: {total_characters}")
            rough_price = tts_provider.estimate_cost(total_characters)
            logger.info(f"Estimate book voiceover would cost you roughly: ${rough_price:.2f}\n")

            # Prompt user to continue if not in preview mode
            if self.config.no_prompt:
                logger.info("Skipping prompt as passed parameter no_prompt")
            elif self.config.preview:
                logger.info("Skipping prompt as in preview mode")
            else:
                confirm_conversion()

            # Prepare chapters for processing
            chapters_to_process = chapters[self.config.chapter_start - 1 : self.config.chapter_end]

            if self.config.chapter_summary:
                logger.info("Generating chapter summaries (this may take a while)...")
                summary_client = _make_summary_client(self.config)
                summary_model = self.config.summary_model or "gpt-4"
                chapters_to_process = list(
                    chapter_summary_iterator(chapters_to_process, summary_client, summary_model)
                )
                tasks = [
                    (idx, title, text, book_parser)
                    for idx, (title, text) in enumerate(chapters_to_process, start=1)
                ]
            else:
                tasks = [
                    (idx, title, text, book_parser)
                    for idx, (title, text) in enumerate(
                        chapters_to_process, start=self.config.chapter_start
                    )
                ]

            # Track failed chapters
            failed_chapters = []
            task_index = {idx: title for idx, title, _, _ in tasks}

            # Use multiprocessing to process chapters in parallel
            with multiprocessing.Pool(
                processes=self.config.worker_count,
                initializer=setup_logging,
                initargs=(self.config.log, self.config.log_file, True)
            ) as pool:
                # Process chapters and collect results
                results = list(pool.imap_unordered(self.process_chapter_wrapper, tasks))

                # Check for failed chapters
                for idx, success in results:
                    if not success:
                        failed_chapters.append((idx, task_index.get(idx, f"chapter-{idx}")))

            if failed_chapters:
                logger.warning("The following chapters failed to convert:")
                for idx, title in failed_chapters:
                    logger.warning(f"  - Chapter {idx}: {title}")
                logger.info(f"Conversion completed with {len(failed_chapters)} failed chapters. Check your output directory: {self.config.output_folder} and log file: {self.config.log_file} for more details.")
            else:
                logger.info(f"All chapters converted successfully. Check your output directory: {self.config.output_folder}")

        except KeyboardInterrupt:
            logger.info("Audiobook generation process interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.exception(f"Error during audiobook generation: {e}")
        finally:
            logger.debug("AudiobookGenerator.run() method finished.")


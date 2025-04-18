import logging
from collections.abc import Iterator, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider

logger = logging.getLogger(__name__)

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://host.docker.internal:8080/v1",
)

MAIN_PROMPT = (
    "{0} Your task is to write a single, concise, "
    "and comprehensive fucking summary that captures all key events and information. Do not use bullet points, "
    "numbered lists, or any symbols such as asterisks (*) for fuck's sake. The summary must be written entirely in full, "
    "well-structured paragraphs, using clear and coherent fucking language throughout. Use an eloquent but still very fucking vulgar, "
    "laidback tone, and use some fucking swear words for good measure, kinda like how "
    "Joe Rogan, Rebecca Ferguson or Fleabag speak."
)


def split_text_iterator(text: str) -> Iterator[str]:
    """Yield split chunks of text for summarization."""
    yield from RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300).split_text(text)


def summarize_chunk(chunk: str) -> str:
    """Generate a summary for a given chunk."""
    summary = invoke(MAIN_PROMPT.format("You are given an excerpt from a book chapter."), chunk)
    logger.debug(f"Chunk summary: {summary}")

    return summary


def chapter_iterator(chapters: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    for title, text in chapters:
        # yield title, text
        yield f"Summary_of_{title}", f"Chapter summary\n\n{summarize_text(text)}"


def invoke(prompt: str, content: str):
    return (
        client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": prompt.strip(),
                },
                {"role": "user", "content": content.strip()},
            ],
        )
        .choices[0]
        .message.content.strip()
    )


def combine_summaries(chunks: Iterable[str]) -> str:
    """Combine multiple summaries into one."""
    summary = invoke(
        MAIN_PROMPT.format("You are given multiple summaries of sections from a book chapter."),
        "\n---\n".join(chunks),
    )
    logger.debug(f"Combined summary: {summary}")

    return summary


def summarize_text(text: str) -> str:
    """Generate a full summary for a chapter."""
    try:
        chunks = tuple(split_text_iterator(text))
        summaries = (summarize_chunk(chunk) for chunk in chunks)

        return combine_summaries(summaries) if len(chunks) > 1 else next(summaries)
    except Exception:
        logger.exception("Failed to generate summary")
        return "Summary not available."


def confirm_conversion() -> None:
    """Ask user to confirm before proceeding."""
    if input("Do you want to continue? (y/n) ").strip().casefold() != "y":
        raise SystemExit("Aborted.")


class AudiobookGenerator:
    def __init__(self, config: GeneralConfig):
        self.config = config

    def process_chapter(self, args: tuple[int, str, str, object, object]) -> None:
        idx, title, text, book_parser, tts_provider = args
        try:
            if self.config.output_text:
                (self.config.output_folder / f"{idx:04d}_{title}.txt").write_text(
                    text, encoding="utf-8"
                )

            if self.config.preview:
                return

            output_path = (
                self.config.output_folder
                / f"{idx:04d}_{title}.{tts_provider.get_output_file_extension()}"
            )
            tags = AudioTags(
                title, book_parser.get_book_author(), book_parser.get_book_title(), idx
            )
            tts_provider.text_to_speech(text, output_path, tags)
        except Exception:
            logger.exception(f"Error processing chapter {idx}")
            raise

    def run(self) -> None:
        try:
            book_parser = get_book_parser(self.config)
            tts_provider = get_tts_provider(self.config)
            self.config.output_folder.mkdir(parents=True, exist_ok=True)

            if self.config.save_cover_image and (cover := book_parser.get_cover()):
                cover_path = self.config.output_folder / f"cover{Path(cover.file_name).suffix}"
                cover_path.write_bytes(cover.get_content())
                logger.info(f"🖼️ Cover image saved as {cover_path.name}")

            chapters = book_parser.get_chapters(tts_provider.get_break_string())

            if not self.config.no_prompt and not self.config.preview:
                confirm_conversion()

            stripped = tuple((title, text) for title, text in chapters if text.strip())
            end = self.config.chapter_end or len(stripped)
            chapters_to_process = stripped[self.config.chapter_start - 1 : end]

            tasks = (
                (idx, title, text, book_parser, tts_provider)
                for idx, (title, text) in enumerate(chapter_iterator(chapters_to_process), start=1)
            )

            with ProcessPoolExecutor(max_workers=self.config.worker_count) as executor:
                for _ in tqdm(
                    as_completed(executor.submit(self.process_chapter, task) for task in tasks),
                    total=len(chapters_to_process),
                    desc="Processing chapters",
                ):
                    pass

            logger.info("All chapters and summaries converted. 🎉")
        except KeyboardInterrupt:
            raise SystemExit("Job stopped by user.")

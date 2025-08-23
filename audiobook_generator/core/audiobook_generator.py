import logging
from collections.abc import Iterator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import math
import openai
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider

logger = logging.getLogger(__name__)

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://host.docker.internal:8080/v1",
)

#  = (
#     "Write a concise, FACTUAL, and comprehensive summary using only the provided section summaries. No opinions. No added context. "
#     "Use full, well-structured paragraphs in clear, coherent, academic language—liberally spiced with profanity. "
#     "Start immediately with the summary. No introductions, no disclaimers, no bullshit."
# )


# FINAL_SUMMARY = (
#     "Your task is to write a concise, FACTUAL and comprehensive summary of the provided text that captures all key events and information. DO NOT provide your opinion. Only work with the information that was provided to you."
#     "The summary must be written entirely in full and well-structured paragraphs. Use an eloquent academic tone, and use a very generous amount of swear words to spice things up. "
#     "Do NOT include any introduction, commentary, or acknowledgment of these instructions. Do NOT use phrases like 'okay, here is your summary' or any bullshit lead-ins. Start immediately with the damn summary."
# )

FINAL_SUMMARY = (
    "Your task is to write a concise, FACTUAL and comprehensive summary of the provided text that captures all key events and information, using ONLY the information provided."
    "The summary must be written entirely in full and well-structured paragraphs. Use an eloquent academic tone, and use a very fucking generous amount of swear words to spice things up. "
    "Do NOT include any introduction, commentary, or acknowledgment of these instructions. Now, get on with the fucking summary."
)

def split_text_iterator(text: str, model: str = "gpt-4o") -> Iterator[str]:
    """Yield split chunks of text guaranteed to fit within 16k context window."""
    enc = tiktoken.encoding_for_model(model)

    # Context window
    max_ctx = 16384
    reserved = 2000   # leave room for system/instructions
    max_tokens_per_chunk = max_ctx - reserved

    # First-pass split by characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,            # conservative, character-based
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )

    for chunk in splitter.split_text(text):
        tokens = enc.encode(chunk)

        # If chunk fits, yield directly
        if len(tokens) <= max_tokens_per_chunk:
            yield chunk
        else:
            # Re-split oversized chunk into smaller safe chunks
            # Roughly target 80% of available tokens as characters
            avg_char_per_token = max(1, len(chunk) // len(tokens))
            safe_char_size = max_tokens_per_chunk * avg_char_per_token * 8 // 10

            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=safe_char_size,
                chunk_overlap=int(safe_char_size * 0.1),
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator="end",
            )
            for sub_chunk in sub_splitter.split_text(chunk):
                # Final guarantee: trim any outliers by tokens
                sub_tokens = enc.encode(sub_chunk)
                if len(sub_tokens) > max_tokens_per_chunk:
                    # hard trim
                    sub_chunk = enc.decode(sub_tokens[:max_tokens_per_chunk])
                yield sub_chunk

def summarize_chunk(chunk: str) -> str:
    """Generate a summary for a given chunk."""
    summary = invoke(FINAL_SUMMARY, chunk)
    logger.debug(f"Chunk summary: {summary}")

    return summary


def chapter_iterator(chapters: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    for title, text in chapters:
        yield title, text
        try:
            summary = summarize_text(text)
            logger.info(summary)
        except Exception as e:
            summary = f"Couldn't generate summary: {e}"

        yield f"Summary_of_{title}", f"Chapter summary\n\n{summary}"


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
    summary = invoke(FINAL_SUMMARY, "\n\n".join(chunks))
    logger.debug(f"Combined summary: {summary}")

    return summary


def summarize_text(text: str) -> str:
    """Generate a full summary for a chapter."""

    chunks = tuple(split_text_iterator(text))

    if len(chunks) == 1:
        return combine_summaries(chunks)
    else:
        return combine_summaries(summarize_chunk(chunk) for chunk in chunks)


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

            with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
                futures = [executor.submit(self.process_chapter, task) for task in tasks]

                for r in as_completed(futures):
                    r.result()

            logger.info("All chapters and summaries converted. 🎉")
        except KeyboardInterrupt:
            raise SystemExit("Job stopped by user.")

import logging
import multiprocessing
from collections.abc import Iterator, Iterable
from pathlib import Path

import openai
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


def split_text_iterator(text: str) -> Iterator[str]:
    """Yield split chunks of text for summarization."""
    yield from RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300).split_text(text)


def summarize_chunk(chunk: str) -> str:
    """Generate a summary for a given chunk."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are given an excerpt from a book chapter. Your task is to write a single, concise, and comprehensive summary that captures all key events and information. Do not use bullet points, numbered lists, or any symbols such as asterisks (*). The summary must be written entirely in full, well-structured paragraphs, using clear and coherent language throughout. Use an eloquent but still laidback tone, and use some fucking swear words for good measure where appropriate, kinda like how Joe Rogan, Rebecca Ferguson, Fleabag or deadmau5 speak.",
            },
            {"role": "user", "content": chunk.strip()},
        ],
    )
    summary = response.choices[0].message.content.strip()
    logger.debug(f"Chunk summary: {summary}")

    return summary


def combine_summaries(chunks: Iterable[str]) -> str:
    """Generate a summary for a given chunk."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are given multiple summaries of sections from a book chapter. Your task is to write a single, concise, and comprehensive summary that captures all key events and information. Do not use bullet points, numbered lists, or any symbols such as asterisks (*). The summary must be written entirely in full, well-structured paragraphs, using clear and coherent language throughout. Use an eloquent but still laidback tone, and use some fucking swear words for good measure words where appropriate, kinda like how Joe Rogan, Rebecca Ferguson, Fleabag or deadmau5 speak.",
            },
            {
                "role": "user",
                "content": "\n\n".join(chunks).strip(),
            },
        ],
    )
    summary = response.choices[0].message.content.strip()
    logger.debug(f"Combined summary: {summary}")

    return summary


def generate_summary(text: str) -> str:
    """Generate a full summary for a chapter."""
    try:
        if len(spl := tuple(split_text_iterator(text))) == 1:
            return summarize_chunk(spl[0])
        return combine_summaries(summarize_chunk(chunk) for chunk in spl)

    except Exception:
        logger.exception("Failed to generate summary")
        return "Summary not available."


def confirm_conversion() -> None:
    """Ask user to confirm before proceeding."""
    if input("Do you want to continue? (y/n) ").strip().lower() != "y":
        print("Aborted.")
        raise SystemExit


class AudiobookGenerator:
    def __init__(self, config: GeneralConfig):
        self.config = config

    def process_chapter(self, args) -> None:
        idx, title, text, book_parser, tts_provider = args
        try:
            logger.info(f"Processing chapter {idx}: {title}")

            if self.config.output_text:
                (self.config.output_folder / f"{idx:04d}_{title}.txt").write_text(text)

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
            logger.info(f"✅ Converted chapter {idx}: {title}")
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
                logger.info("🖼️ Cover image saved as %s", cover_path.name)

            chapters = book_parser.get_chapters(tts_provider.get_break_string())

            if not self.config.no_prompt and not self.config.preview:
                confirm_conversion()

            tasks = (
                (idx, title, text, book_parser, tts_provider)
                for idx, (title, text) in enumerate(self.chapter_iterator(chapters), start=1)
            )

            with multiprocessing.Pool(self.config.worker_count) as pool:
                for _ in pool.imap_unordered(self.process_chapter, tasks):
                    pass

            logger.info("All chapters and summaries converted. 🎉")
        except KeyboardInterrupt:
            logger.info("Job stopped by user.")
            raise SystemExit

    def chapter_iterator(self, chapters) -> Iterator[tuple[str, str]]:
        filtered = [(title, text) for title, text in chapters if text.strip()]
        logger.info("Chapters count: %d.", len(filtered))

        end = self.config.chapter_end or len(filtered)
        for title, text in filtered[self.config.chapter_start - 1 : end]:
            yield title, text
            yield f"Summary_of_{title}", f"Chapter summary\n\n{generate_summary(text)}"

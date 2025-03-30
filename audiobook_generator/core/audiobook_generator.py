import logging
import multiprocessing
from pathlib import Path

import openai

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider

logger = logging.getLogger(__name__)


def generate_summary(text):
    """Generate a summary using OpenAI's GPT model."""
    try:
        client = openai.OpenAI(
            api_key="your-api-key",  # Replace with your actual API key
            base_url="http://host.docker.internal:8080/v1",  # Custom API base URL
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are given a book chapter. Write a summary in a concise and comprehensive way, ensuring it covers all the key events and information presented. Your goal is to provide enough detail about the characters, the events and so forth, that someone who didn't fully pay attention to the chapter could understand what happened and feel prepared to continue reading. Stick to the facts presented in the summary and avoid any speculation, analysis, or personal opinions. Do not use bullet points, the output should be formatted as regular paragraphs.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Summary not available."


def confirm_conversion():
    print("Do you want to continue? (y/n)")
    answer = input()
    if answer.lower() != "y":
        print("Aborted.")
        exit(0)


def get_total_chars(chapters):
    return sum(len(text) for _, text in chapters)


class AudiobookGenerator:
    def __init__(self, config: GeneralConfig):
        self.config = config

    def process_chapter(self, args):
        idx, title, text, book_parser, tts_provider = args
        try:
            logger.info(f"Processing chapter {idx}: {title}")

            if self.config.output_text:
                text_file = self.config.output_folder / f"{idx:04d}_{title}.txt"
                text_file.write_text(text, encoding="utf-8")

            if self.config.preview:
                return

            output_file = (
                self.config.output_folder
                / f"{idx:04d}_{title}.{tts_provider.get_output_file_extension()}"
            )
            audio_tags = AudioTags(
                title, book_parser.get_book_author(), book_parser.get_book_title(), idx
            )
            tts_provider.text_to_speech(text, output_file, audio_tags)
            logger.info(f"✅ Converted chapter {idx}: {title}")
        except Exception:
            logger.exception(f"Error processing chapter {idx}")
            raise

    def run(self):
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

            with multiprocessing.Pool(processes=self.config.worker_count) as pool:
                for _ in pool.imap_unordered(self.process_chapter, tasks):
                    pass

            logger.info("All chapters and summaries converted. 🎉")
        except KeyboardInterrupt:
            logger.info("Job stopped by user.")
            exit()

    def chapter_iterator(self, chapters):
        chapters = [(title, text) for title, text in chapters if text.strip()]
        logger.info(f"Chapters count: {len(chapters)}.")
        if self.config.chapter_end == -1:
            self.config.chapter_end = len(chapters)
        chapters_to_process = chapters[self.config.chapter_start - 1 : self.config.chapter_end]

        for chapter in chapters_to_process:
            title, text = chapter

            yield title, text
            yield f"Summary_of_{title}", "Chapter summary\n\n" + generate_summary(text)

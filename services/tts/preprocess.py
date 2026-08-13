"""
services/tts/preprocess.py

Text preprocessing and normalization pipeline for the Valora AI
Text-to-Speech service.

This module prepares raw user-supplied text for synthesis by:
    - Cleaning and normalizing whitespace, punctuation, and quotes.
    - Stripping unsupported/control Unicode characters while preserving
      multilingual scripts (Latin, Devanagari, Japanese, etc.).
    - Optionally removing emojis.
    - Expanding common abbreviations.
    - Expanding standalone numbers into spoken form (English).
    - Detecting degenerate input (empty text, pathological repeated
      characters, oversized input).
    - Splitting text into sentence-aware chunks that respect
      `settings.TTS_MAX_CHUNK_CHARS` and `settings.TTS_MAX_TEXT_LENGTH`.

The output of `TextPreprocessor.preprocess()` is a `PreprocessedText`
whose `.chunks` list is intended to be fed directly into
`services/tts/inference.py` (e.g. one `SynthesisRequest` per chunk, or
by reusing `TTSInferenceService._chunk_text` semantics upstream).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Pattern

from services.tts.config import settings

logger = logging.getLogger("valora.tts.preprocess")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TextValidationError(Exception):
    """Raised when input text fails validation prior to synthesis."""


@dataclass
class PreprocessedText:
    """
    Result of running text through the preprocessing pipeline.

    Attributes:
        original_text: The raw, unmodified input text.
        normalized_text: The fully cleaned and normalized text.
        chunks: Ordered list of synthesis-ready text chunks, each no
            longer than `settings.TTS_MAX_CHUNK_CHARS`.
        char_count: Length of `normalized_text`, in characters.
        chunk_count: Number of chunks produced.
    """

    original_text: str
    normalized_text: str
    chunks: List[str] = field(default_factory=list)
    char_count: int = 0
    chunk_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.normalized_text)
        self.chunk_count = len(self.chunks)


class TextPreprocessor:
    """
    Cleans, normalizes, validates, and chunks text prior to TTS
    synthesis, with multilingual support for English, Hindi, French,
    Japanese, and Spanish.

    Instances are stateless and thread-safe; a single instance may be
    shared across concurrent requests.

    Example:
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Dr. Smith arrived at 3pm!!!")
        for chunk in result.chunks:
            ...  # feed to TTSInferenceService
    """

    # -------------------------------------------------------------- #
    # Class-level compiled patterns (built once, reused across calls)
    # -------------------------------------------------------------- #

    _WHITESPACE_PATTERN: Pattern[str] = re.compile(r"\s+")

    _CONTROL_CHARS_PATTERN: Pattern[str] = re.compile(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
    )

    # Allowed Unicode blocks: Latin (+ extended/supplement), common
    # punctuation/symbols, Devanagari (Hindi), Japanese (Hiragana,
    # Katakana, Kanji/CJK), Spanish/French accented Latin, digits, and
    # general whitespace/punctuation.
    _ALLOWED_CHAR_PATTERN: Pattern[str] = re.compile(
        r"["
        r"\u0000-\u024F"          # Basic Latin, Latin-1, Latin Extended-A/B
        r"\u0300-\u036F"          # Combining diacritical marks
        r"\u0900-\u097F"          # Devanagari (Hindi)
        r"\u3000-\u303F"          # CJK punctuation
        r"\u3040-\u309F"          # Hiragana
        r"\u30A0-\u30FF"          # Katakana
        r"\u4E00-\u9FFF"          # CJK Unified Ideographs (Kanji)
        r"\uFF00-\uFFEF"          # Fullwidth forms
        r"\u2000-\u206F"          # General punctuation
        r"\u20A0-\u20CF"          # Currency symbols
        r"\s"
        r"]"
    )

    _EMOJI_PATTERN: Pattern[str] = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "\U00002190-\U000021FF"
        "\U00002B00-\U00002BFF"
        "\U0000FE0F"
        "]+",
        flags=re.UNICODE,
    )

    _REPEATED_CHAR_PATTERN: Pattern[str] = re.compile(r"(.)\1{4,}", flags=re.UNICODE)

    _SENTENCE_SPLIT_PATTERN: Pattern[str] = re.compile(
        r"(?<=[.!?。！？])\s+"
    )

    _QUOTE_MAP = {
        "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
        "\u00AB": '"', "\u00BB": '"',
    }

    _DASH_MAP = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-",
        "\u2013": "-", "\u2014": "-", "\u2015": "-",
    }

    _ABBREVIATIONS = {
        "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss",
        "Prof.": "Professor", "Sr.": "Senior", "Jr.": "Junior",
        "St.": "Saint", "vs.": "versus", "etc.": "et cetera",
        "e.g.": "for example", "i.e.": "that is", "approx.": "approximately",
        "dept.": "department", "govt.": "government", "no.": "number",
    }

    _ABBREVIATION_PATTERN: Pattern[str] = re.compile(
        r"(?<![\w])(" + "|".join(re.escape(k) for k in _ABBREVIATIONS) + r")(?![\w])"
    )

    _NUMBER_PATTERN: Pattern[str] = re.compile(r"(?<![\w.])\d{1,15}(?![\w])")

    _ONES = (
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    )
    _TENS = (
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
    )
    _SCALES = ("", "thousand", "million", "billion", "trillion")

    def __init__(
        self,
        max_text_length: Optional[int] = None,
        max_chunk_chars: Optional[int] = None,
        remove_emojis: bool = True,
        expand_numbers: bool = True,
        expand_abbreviations: bool = True,
    ) -> None:
        """
        Initialize the text preprocessor.

        Args:
            max_text_length: Maximum allowed total input length, in
                characters. Defaults to `settings.TTS_MAX_TEXT_LENGTH`.
            max_chunk_chars: Maximum characters per output chunk.
                Defaults to `settings.TTS_MAX_CHUNK_CHARS`.
            remove_emojis: Whether emojis should be stripped from text.
                If False, emojis are left in place (not spoken).
            expand_numbers: Whether standalone integers should be
                expanded into spoken English words.
            expand_abbreviations: Whether known abbreviations should be
                expanded into their full spoken form.
        """
        self.max_text_length: int = max_text_length or settings.TTS_MAX_TEXT_LENGTH
        self.max_chunk_chars: int = max_chunk_chars or settings.TTS_MAX_CHUNK_CHARS
        self.remove_emojis: bool = remove_emojis
        self.expand_numbers: bool = expand_numbers
        self.expand_abbreviations: bool = expand_abbreviations

        if self.max_chunk_chars > self.max_text_length:
            logger.warning(
                "max_chunk_chars (%d) exceeds max_text_length (%d); "
                "clamping max_chunk_chars.",
                self.max_chunk_chars,
                self.max_text_length,
            )
            self.max_chunk_chars = self.max_text_length

        logger.info(
            "TextPreprocessor initialized (max_text_length=%d, max_chunk_chars=%d, "
            "remove_emojis=%s, expand_numbers=%s, expand_abbreviations=%s).",
            self.max_text_length,
            self.max_chunk_chars,
            self.remove_emojis,
            self.expand_numbers,
            self.expand_abbreviations,
        )

    # -------------------------------------------------------------- #
    # Validation
    # -------------------------------------------------------------- #

    def validate_text(self, text: str) -> None:
        """
        Validate raw input text prior to any processing.

        Args:
            text: The raw input text to validate.

        Raises:
            TextValidationError: If the text is not a string, is empty
                or whitespace-only, exceeds `max_text_length`, or
                contains a pathological run of repeated characters.
        """
        if not isinstance(text, str):
            raise TextValidationError(f"Input must be a string, got {type(text).__name__}.")

        if not text.strip():
            raise TextValidationError("Input text is empty or whitespace-only.")

        if len(text) > self.max_text_length:
            raise TextValidationError(
                f"Input text exceeds maximum length of {self.max_text_length} "
                f"characters (got {len(text)})."
            )

        match = self._REPEATED_CHAR_PATTERN.search(text)
        if match and len(match.group(0)) > 40:
            raise TextValidationError(
                "Input text contains an excessively long repeated character "
                f"sequence ('{match.group(1)}' x {len(match.group(0))})."
            )

    # -------------------------------------------------------------- #
    # Cleaning
    # -------------------------------------------------------------- #

    def clean_text(self, text: str) -> str:
        """
        Remove control characters, disallowed Unicode code points, and
        (optionally) emojis, while preserving supported scripts.

        Args:
            text: Raw or partially processed input text.

        Returns:
            str: Text with unsupported characters removed.
        """
        text = unicodedata.normalize("NFC", text)
        text = self._CONTROL_CHARS_PATTERN.sub("", text)

        if self.remove_emojis:
            text = self._EMOJI_PATTERN.sub("", text)

        text = "".join(
            char for char in text if self._ALLOWED_CHAR_PATTERN.match(char)
        )

        return text

    # -------------------------------------------------------------- #
    # Normalization
    # -------------------------------------------------------------- #

    def normalize_text(self, text: str) -> str:
        """
        Normalize whitespace, quotes, dashes, punctuation, numbers,
        and abbreviations in already-cleaned text.

        Args:
            text: Cleaned input text (see `clean_text`).

        Returns:
            str: Fully normalized text ready for chunking.
        """
        for src, dst in self._QUOTE_MAP.items():
            text = text.replace(src, dst)
        for src, dst in self._DASH_MAP.items():
            text = text.replace(src, dst)

        text = self._normalize_repeated_chars(text)
        text = self._normalize_punctuation(text)

        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)

        if self.expand_numbers:
            text = self._expand_numbers(text)

        text = self._WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def _normalize_repeated_chars(self, text: str) -> str:
        """
        Collapse runs of 3+ identical non-space characters down to a
        maximum of 3 repetitions, to avoid degenerate synthesis input
        (e.g. "soooooo" -> "sooo").

        Args:
            text: Input text.

        Returns:
            str: Text with excessive character repetition collapsed.
        """
        return re.sub(r"(\S)\1{3,}", r"\1\1\1", text)

    def _normalize_punctuation(self, text: str) -> str:
        """
        Normalize repeated punctuation (e.g. "!!!" -> "!") and ensure
        consistent spacing around common punctuation marks.

        Args:
            text: Input text.

        Returns:
            str: Text with normalized punctuation.
        """
        text = re.sub(r"([!?.,;:])\1{1,}", r"\1", text)
        text = re.sub(r"\s+([!?.,;:])", r"\1", text)
        text = re.sub(r"([!?.,;:])(?=[^\s])", r"\1 ", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Replace known abbreviations with their full spoken form.

        Args:
            text: Input text.

        Returns:
            str: Text with recognized abbreviations expanded.
        """
        return self._ABBREVIATION_PATTERN.sub(
            lambda m: self._ABBREVIATIONS[m.group(1)], text
        )

    def _expand_numbers(self, text: str) -> str:
        """
        Expand standalone integer sequences into spoken English words.
        Non-numeric or overly large numbers are left unchanged.

        Args:
            text: Input text.

        Returns:
            str: Text with eligible numbers expanded.
        """
        def _replace(match: "re.Match[str]") -> str:
            token = match.group(0)
            try:
                return self._number_to_words(int(token))
            except (ValueError, OverflowError):
                return token

        return self._NUMBER_PATTERN.sub(_replace, text)

    def _number_to_words(self, number: int) -> str:
        """
        Convert an integer into spoken English words.

        Args:
            number: The integer to convert. Must fit within the
                supported scale (up to trillions).

        Returns:
            str: The spoken-word representation of the number.

        Raises:
            OverflowError: If the number exceeds the supported range.
        """
        if number == 0:
            return "zero"
        if number >= 10 ** (3 * len(self._SCALES)):
            raise OverflowError("Number too large to expand.")

        negative = number < 0
        number = abs(number)

        groups: List[int] = []
        remainder = number
        while remainder > 0:
            groups.append(remainder % 1000)
            remainder //= 1000

        words: List[str] = []
        for idx in range(len(groups) - 1, -1, -1):
            group_value = groups[idx]
            if group_value == 0:
                continue
            group_words = self._three_digit_to_words(group_value)
            scale = self._SCALES[idx] if idx < len(self._SCALES) else ""
            words.append(f"{group_words} {scale}".strip())

        result = " ".join(words)
        return f"negative {result}" if negative else result

    def _three_digit_to_words(self, value: int) -> str:
        """
        Convert an integer in the range [0, 999] into spoken English
        words.

        Args:
            value: Integer between 0 and 999 inclusive.

        Returns:
            str: Spoken-word representation of the value.
        """
        parts: List[str] = []
        hundreds, remainder = divmod(value, 100)

        if hundreds:
            parts.append(f"{self._ONES[hundreds]} hundred")

        if remainder:
            if remainder < 20:
                parts.append(self._ONES[remainder])
            else:
                tens, ones = divmod(remainder, 10)
                tens_word = self._TENS[tens]
                parts.append(f"{tens_word}-{self._ONES[ones]}" if ones else tens_word)

        return " ".join(parts)

    # -------------------------------------------------------------- #
    # Sentence splitting and chunking
    # -------------------------------------------------------------- #

    def split_sentences(self, text: str) -> List[str]:
        """
        Split normalized text into sentences using punctuation-based
        boundaries that support Latin, Devanagari, and CJK terminators.

        Args:
            text: Normalized input text.

        Returns:
            List[str]: Ordered list of non-empty sentences.
        """
        if not text:
            return []

        sentences = self._SENTENCE_SPLIT_PATTERN.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """
        Split normalized text into chunks no longer than
        `max_chunk_chars`, preferring sentence boundaries and falling
        back to hard splits for oversized sentences.

        Args:
            text: Normalized input text.

        Returns:
            List[str]: Ordered list of non-empty text chunks.
        """
        if not text:
            return []

        if len(text) <= self.max_chunk_chars:
            return [text]

        sentences = self.split_sentences(text)
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            if len(sentence) > self.max_chunk_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._hard_split(sentence))
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > self.max_chunk_chars:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    def _hard_split(self, text: str) -> List[str]:
        """
        Split an oversized sentence into fixed-size chunks, breaking on
        whitespace where possible to avoid splitting words mid-token.

        Args:
            text: A single sentence longer than `max_chunk_chars`.

        Returns:
            List[str]: Ordered list of chunk strings.
        """
        chunks: List[str] = []
        remaining = text.strip()

        while len(remaining) > self.max_chunk_chars:
            window = remaining[: self.max_chunk_chars]
            split_at = window.rfind(" ")
            if split_at <= 0:
                split_at = self.max_chunk_chars

            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    # -------------------------------------------------------------- #
    # Orchestration
    # -------------------------------------------------------------- #

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Run the full preprocessing pipeline: validate, clean, normalize,
        and chunk the input text.

        Args:
            text: Raw input text to prepare for synthesis.

        Returns:
            PreprocessedText: The original text alongside normalized
                text and synthesis-ready chunks.

        Raises:
            TextValidationError: If the input fails validation at any
                stage of the pipeline.
        """
        self.validate_text(text)

        cleaned = self.clean_text(text)
        if not cleaned.strip():
            raise TextValidationError(
                "Input text contained no supported characters after cleaning."
            )

        normalized = self.normalize_text(cleaned)
        if not normalized.strip():
            raise TextValidationError(
                "Input text was empty after normalization."
            )

        if len(normalized) > self.max_text_length:
            logger.warning(
                "Normalized text (%d chars) exceeds max_text_length (%d); truncating.",
                len(normalized),
                self.max_text_length,
            )
            normalized = normalized[: self.max_text_length].strip()

        chunks = self.chunk_text(normalized)
        if not chunks:
            raise TextValidationError("No valid chunks could be produced from input text.")

        logger.info(
            "Preprocessed text: %d char(s) -> %d chunk(s).",
            len(normalized),
            len(chunks),
        )

        return PreprocessedText(
            original_text=text,
            normalized_text=normalized,
            chunks=chunks,
        )
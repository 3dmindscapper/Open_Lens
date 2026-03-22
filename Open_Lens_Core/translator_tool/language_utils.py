"""
Language detection and translation utilities.

Language detection: langdetect  (offline, statistical)
Translation:        argostranslate (fully offline neural MT — models downloaded once,
                    then run locally with no internet connection required)
                    Optional: Ollama (local LLM server) for domain-specific quality
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language code mappings
# ---------------------------------------------------------------------------

# langdetect → Tesseract language pack names
LANGDETECT_TO_TESSERACT: Dict[str, str] = {
    "af": "afr",
    "ar": "ara",
    "az": "aze",
    "be": "bel",
    "bg": "bul",
    "bs": "bos",
    "ca": "cat",
    "cs": "ces",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "eo": "epo",
    "es": "spa",
    "et": "est",
    "eu": "eus",
    "fa": "fas",
    "fi": "fin",
    "fr": "fra",
    "ga": "gle",
    "gl": "glg",
    "gu": "guj",
    "he": "heb",
    "hi": "hin",
    "hr": "hrv",
    "hu": "hun",
    "hy": "hye",
    "id": "ind",
    "is": "isl",
    "it": "ita",
    "ja": "jpn",
    "ka": "kat",
    "kn": "kan",
    "ko": "kor",
    "lt": "lit",
    "lv": "lav",
    "mk": "mkd",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "msa",
    "mt": "mlt",
    "ne": "nep",
    "nl": "nld",
    "no": "nor",
    "pa": "pan",
    "pl": "pol",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sk": "slk",
    "sl": "slv",
    "sq": "sqi",
    "sr": "srp",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tl": "tgl",
    "tr": "tur",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "zh": "chi_sim",
}

# Languages written right-to-left
RTL_LANGUAGES = {"ar", "he", "fa", "ur", "yi", "dv", "ku", "ps"}


def langdetect_to_tesseract(lang_code: str) -> str:
    """Map a langdetect language code to a Tesseract language pack name."""
    return LANGDETECT_TO_TESSERACT.get(lang_code.lower(), "eng")


def is_rtl(lang_code: str) -> bool:
    """Return True if the language is written right-to-left."""
    return lang_code.lower().split("-")[0] in RTL_LANGUAGES


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect the language of *text* and return an ISO 639-1 code.

    Falls back to ``"en"`` if detection fails or the text is too short.

    Args:
        text: A representative sample of text (ideally ≥ 50 characters).

    Returns:
        ISO 639-1 language code string, e.g. ``"en"``, ``"fr"``, ``"zh-cn"``.
    """
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        from langdetect import detect, DetectorFactory  # type: ignore
        DetectorFactory.seed = 0  # reproducible results
        return detect(text)
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Translation  (fully offline via Argos Translate)
# ---------------------------------------------------------------------------
# Python 3.14 fix: spacy → pydantic v1 crashes with
# "unable to infer type for attribute REGEX".
# We mock spacy (never needed) and force MiniSBD sentencizer instead of stanza.
# ---------------------------------------------------------------------------

def _install_argostranslate_mocks():
    """Prevent spacy/stanza from being imported (pydantic v1 crash on Python 3.14).
    Force MiniSBD sentence boundary detection which works without either."""
    import sys
    import types

    # Mock spacy (causes pydantic crash)
    if "spacy" not in sys.modules:
        _spacy = types.ModuleType("spacy")
        _spacy.load = lambda *a, **kw: None
        sys.modules["spacy"] = _spacy

    # Mock stanza (also triggers pydantic via confection)
    if "stanza" not in sys.modules:
        sys.modules["stanza"] = types.ModuleType("stanza")

    # Force MiniSBD so argostranslate never tries stanza/spacy at runtime
    try:
        from argostranslate import settings
        settings.chunk_type = settings.ChunkType.MINISBD
    except Exception:
        pass


import sys as _sys
import types as _types
_install_argostranslate_mocks()


def _require_argostranslate():
    try:
        _install_argostranslate_mocks()
        from argostranslate import package as _pkg, translate as _tr  # type: ignore
        return _pkg, _tr
    except ImportError:
        raise ImportError(
            "argostranslate is required for local offline translation.\n"
            "Install it with:  pip install argostranslate\n"
            "Language models are downloaded automatically on first use."
        )


def _normalise_lang(code: str) -> str:
    """Map extended codes like 'zh-cn' / 'zh-tw' to Argos-Translate codes."""
    mapping = {
        "zh-cn": "zh",
        "zh-tw": "zt",
        "zh": "zh",
    }
    return mapping.get(code.lower(), code.lower().split("-")[0])


def _ensure_language_pair(from_code: str, to_code: str, log=print) -> bool:
    """Download the Argos Translate model for *from_code* → *to_code* if needed.
    Returns True if the pair is (now) available, False otherwise.
    """
    _pkg, _tr = _require_argostranslate()

    # --- check already installed (no pydantic) ---
    try:
        installed = _tr.get_installed_languages()
        installed_codes = {lang.code for lang in installed}
        if from_code in installed_codes and to_code in installed_codes:
            from_lang = next((l for l in installed if l.code == from_code), None)
            if from_lang:
                t = from_lang.get_translation(
                    next((l for l in installed if l.code == to_code), None)
                )
                if t is not None:
                    return True
    except Exception:
        pass

    log(f"    [translate] Model {from_code}→{to_code} not found locally. Downloading (~100 MB) …")

    # --- Attempt 1: argostranslate's own package API ---
    try:
        _pkg.update_package_index()
        available = _pkg.get_available_packages()
        matched = [p for p in available if p.from_code == from_code and p.to_code == to_code]
        if matched:
            download_path = matched[0].download()
            _pkg.install_from_path(download_path)
            log(f"    [translate] Model installed: {from_code}→{to_code}")
            return True
        log(f"    [translate] No package in index for {from_code}→{to_code}")
        return False
    except Exception as e1:
        log(f"    [translate] Standard download failed ({e1}); trying urllib fallback …")

    # --- Attempt 2: download the index via plain urllib (no pydantic) ---
    try:
        import urllib.request
        import json as _json
        import tempfile
        import os as _os

        _INDEX_URLS = [
            "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json",
            "https://argospm-index.s3.amazonaws.com/index.json",
        ]
        index = None
        for idx_url in _INDEX_URLS:
            try:
                with urllib.request.urlopen(idx_url, timeout=30) as resp:
                    index = _json.loads(resp.read().decode("utf-8"))
                break
            except Exception:
                continue

        if index is None:
            log("    [translate] Could not retrieve package index. Install models manually.")
            return False

        pkg_info = next(
            (p for p in index if p.get("from_code") == from_code and p.get("to_code") == to_code),
            None,
        )
        if pkg_info is None:
            log(f"    [translate] Package {from_code}→{to_code} not found in index.")
            return False

        links = pkg_info.get("links", [])
        if not links:
            return False

        url = links[0]
        log(f"    [translate] Downloading {url} …")
        tmp_file = _os.path.join(
            tempfile.gettempdir(), f"argos_{from_code}_{to_code}.argosmodel"
        )
        urllib.request.urlretrieve(url, tmp_file)
        _pkg.install_from_path(tmp_file)
        log(f"    [translate] Model installed: {from_code}→{to_code}")
        return True
    except Exception as e2:
        log(f"    [translate] urllib fallback also failed: {e2}")
        return False


def _get_translation_fn(from_code: str, to_code: str, log=print):
    """Return a callable translate(text) → str, or None if unavailable."""
    _pkg, _tr = _require_argostranslate()

    # Direct pair
    if _ensure_language_pair(from_code, to_code, log=log):
        try:
            installed = _tr.get_installed_languages()
            from_lang = next((l for l in installed if l.code == from_code), None)
            to_lang = next((l for l in installed if l.code == to_code), None)
            if from_lang and to_lang:
                translation = from_lang.get_translation(to_lang)
                if translation:
                    return translation.translate
        except Exception as exc:
            log(f"    [translate] Error getting translation object: {exc}")

    # Pivot through English (e.g. ro → en → ja)
    if from_code != "en" and to_code != "en":
        ok1 = _ensure_language_pair(from_code, "en", log=log)
        ok2 = _ensure_language_pair("en", to_code, log=log)
        if ok1 and ok2:
            try:
                installed = _tr.get_installed_languages()
                from_lang = next((l for l in installed if l.code == from_code), None)
                en_lang   = next((l for l in installed if l.code == "en"),       None)
                to_lang   = next((l for l in installed if l.code == to_code),    None)
                if from_lang and en_lang and to_lang:
                    t1 = from_lang.get_translation(en_lang)
                    t2 = en_lang.get_translation(to_lang)
                    if t1 and t2:
                        def _pivot(text, _t1=t1, _t2=t2):
                            return _t2.translate(_t1.translate(text))
                        return _pivot
            except Exception as exc:
                log(f"    [translate] Pivot setup error: {exc}")

    return None


# Characters commonly used as list bullets / checkmarks in documents.
# These are preserved as-is during translation so they aren't lost or
# corrupted by the translation engine.
_BULLET_RE = re.compile(
    r'^(\s*[\u2713\u2714\u2715\u2716\u2717\u2718'
    r'\u2610\u2611\u2612\u2022\u25cf\u25cb\u25a0\u25a1'
    r'\u25b6\u25b8\u25b9\u25bb\u27a4\u279c\u2794\u2192\u21d2'
    r'\u2219\u00b7\u2043\u2013\u2014\u2023\u2039\u203a'
    r'\u25aa\u25ab\u29bf\u2756\*\-]\s*)+'
)


def _split_bullet_prefix(text: str):
    """Split leading bullet / checkmark characters from the translatable body."""
    m = _BULLET_RE.match(text)
    if m:
        return m.group(0), text[m.end():]
    return "", text


# ---------------------------------------------------------------------------
# Block-level data classification
# ---------------------------------------------------------------------------
# Instead of trying to protect individual tokens within a phrase (which
# fragments translatable text and causes partial-word inpainting), we
# classify entire blocks as either "data-only" or "translatable".
#
# A data-only block contains no real words — just numbers, dates, codes,
# punctuation, and symbols.  These are never sent to the translation
# engine and their original pixels are preserved.
# ---------------------------------------------------------------------------

# Match sequences of 2+ consecutive alphabetic characters (including accented)
_WORD_RE = re.compile(r'[a-zA-Z\u00C0-\u024F]{2,}')


def is_data_only_block(text: str) -> bool:
    """Return True if *text* contains no meaningful translatable words.

    Blocks that are purely numeric (dates, codes, amounts, postal codes)
    should not be translated or inpainted — their original pixels are
    already correct.

    Conservative approach: a block is data-only ONLY if it contains no
    word-like sequences of 2+ alphabetic characters.  Mixed blocks like
    ``"Le 14/09/1964 à Metz (57)"`` contain words ("Le", "Metz") and
    will be translated so no untranslated text remains visible.
    """
    if not text:
        return True
    words = _WORD_RE.findall(text)
    return len(words) == 0


def _post_process_translation(text: str, source_lang: str, target_lang: str) -> str:
    """Clean up translation artefacts (whitespace, punctuation).

    No hardcoded word replacements — the translation engine output is
    trusted as-is so the tool stays flexible across all document types.
    """
    if not text:
        return text

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text).strip()

    return text


def translate_blocks(
    blocks: List[Dict[str, Any]],
    target_lang: str,
    source_lang: str = "auto",
    log=print,
    **_kwargs,
) -> List[Dict[str, Any]]:
    """Translate each block's ``text`` offline using Argos Translate.
    Models (~100 MB) are downloaded once on first use, then cached to disk.
    """
    if not blocks:
        return blocks

    from_code = _normalise_lang(source_lang) if source_lang not in ("auto", "") else "en"
    to_code   = _normalise_lang(target_lang)

    if from_code == to_code:
        for block in blocks:
            block["translated_text"] = block["text"]
        return blocks

    try:
        translate_fn = _get_translation_fn(from_code, to_code, log=log)
    except Exception as exc:
        log(f"    [translate] Cannot load translation engine: {exc}")
        translate_fn = None

    if translate_fn is None:
        log(f"    [translate] No model available for {from_code}→{to_code}. Text kept as-is.")
        for block in blocks:
            block["translated_text"] = block["text"]
        return blocks

    # Classify blocks: data-only blocks are preserved as-is.
    data_count = 0
    translatable_blocks = []
    for block in blocks:
        if is_data_only_block(block["text"]):
            block["translated_text"] = block["text"]
            block["_data_only"] = True
            data_count += 1
        else:
            translatable_blocks.append(block)

    if data_count:
        log(f"    [translate] {data_count} data-only block(s) preserved as-is.")

    log(f"    [translate] Translating {len(translatable_blocks)} block(s) …")
    for block in translatable_blocks:
        try:
            prefix, body = _split_bullet_prefix(block["text"])
            if body.strip():
                result = translate_fn(body)
                if result:
                    result = _post_process_translation(result, from_code, to_code)
                    block["translated_text"] = prefix + result
                else:
                    block["translated_text"] = block["text"]
            else:
                block["translated_text"] = block["text"]
        except Exception as exc:
            log(f"    [translate] Block failed: {exc}")
            block["translated_text"] = block["text"]

    return blocks


def translate_single(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Translate a single string locally using Argos Translate."""
    from_code = _normalise_lang(source_lang) if source_lang not in ("auto", "") else "en"
    to_code   = _normalise_lang(target_lang)
    if from_code == to_code:
        return text
    fn = _get_translation_fn(from_code, to_code)
    if fn is None:
        return text
    try:
        return fn(text) or text
    except Exception as exc:
        print(f"  [translate] Single translation failed: {exc}")
        return text


# ---------------------------------------------------------------------------
# Ollama-based translation (optional local LLM)
# ---------------------------------------------------------------------------

def _translate_ollama(
    text: str,
    source_lang: str,
    target_lang: str,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b",
    context: str = "",
    timeout: int = 60,
) -> Optional[str]:
    """Translate *text* using a local Ollama LLM server.

    Returns the translated text or ``None`` if the server is unreachable.
    """
    import urllib.request
    import json

    # Build a structured prompt that constrains the model to return only
    # the translation, with optional page-level context for coherence.
    ctx_part = ""
    if context:
        ctx_part = (
            f"\nFor context, here is the full page text (same language as the source):\n"
            f"---\n{context[:2000]}\n---\n"
        )

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        f"Return ONLY the translated text, nothing else. "
        f"Preserve formatting, numbers, dates, and proper nouns as-is."
        f"{ctx_part}\n"
        f"Text to translate:\n{text}"
    )

    payload = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip() or None
    except Exception as exc:
        log.warning("Ollama translation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Glossary support
# ---------------------------------------------------------------------------

_GLOSSARY_CACHE: Dict[str, Dict[str, str]] = {}


def _load_glossary(glossary_path: str) -> Dict[str, str]:
    """Load a TSV glossary file (source_term<TAB>target_term) into a dict."""
    if glossary_path in _GLOSSARY_CACHE:
        return _GLOSSARY_CACHE[glossary_path]

    glossary: Dict[str, str] = {}
    try:
        with open(glossary_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    glossary[parts[0].strip()] = parts[1].strip()
    except Exception as exc:
        log.warning("Failed to load glossary %s: %s", glossary_path, exc)

    _GLOSSARY_CACHE[glossary_path] = glossary
    return glossary


def _apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    """Replace glossary source terms with target terms in translated text."""
    for source_term, target_term in glossary.items():
        # Case-insensitive whole-word replacement
        pattern = re.compile(re.escape(source_term), re.IGNORECASE)
        text = pattern.sub(target_term, text)
    return text


# ---------------------------------------------------------------------------
# Context-aware batch translation
# ---------------------------------------------------------------------------

def translate_blocks_batch(
    blocks: List[Dict[str, Any]],
    target_lang: str,
    source_lang: str = "auto",
    engine: str = "argos",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b",
    glossary_path: Optional[str] = None,
    log_fn=print,
    **_kwargs,
) -> List[Dict[str, Any]]:
    """Translate blocks with optional context awareness and engine selection.

    This is the new unified entry point that replaces ``translate_blocks``
    when called from the upgraded pipeline.

    Args:
        blocks:          Text blocks from OCR.
        target_lang:     Target language code.
        source_lang:     Source language code or ``"auto"``.
        engine:          ``"argos"`` or ``"ollama"``.
        ollama_url:      Ollama server URL.
        ollama_model:    Ollama model name.
        glossary_path:   Path to a TSV glossary file (optional).
        log_fn:          Logger callable.

    Returns:
        Blocks with ``translated_text`` populated.
    """
    if not blocks:
        return blocks

    from_code = _normalise_lang(source_lang) if source_lang not in ("auto", "") else "en"
    to_code = _normalise_lang(target_lang)

    if from_code == to_code:
        for block in blocks:
            block["translated_text"] = block["text"]
        return blocks

    # Classify data-only blocks
    data_count = 0
    translatable_blocks = []
    for block in blocks:
        if is_data_only_block(block["text"]):
            block["translated_text"] = block["text"]
            block["_data_only"] = True
            data_count += 1
        else:
            translatable_blocks.append(block)

    if data_count:
        log_fn(f"    [translate] {data_count} data-only block(s) preserved as-is.")

    if not translatable_blocks:
        return blocks

    # Build page-level context for coherence
    page_context = " ".join(b["text"] for b in translatable_blocks)

    # Load glossary if provided
    glossary = _load_glossary(glossary_path) if glossary_path else {}

    # --- Ollama engine ---
    if engine == "ollama":
        log_fn(f"    [translate] Using Ollama ({ollama_model}) for {len(translatable_blocks)} block(s)")
        for block in translatable_blocks:
            prefix, body = _split_bullet_prefix(block["text"])
            if body.strip():
                result = _translate_ollama(
                    body, source_lang, target_lang,
                    ollama_url=ollama_url, ollama_model=ollama_model,
                    context=page_context,
                )
                if result:
                    result = _post_process_translation(result, from_code, to_code)
                    if glossary:
                        result = _apply_glossary(result, glossary)
                    block["translated_text"] = prefix + result
                else:
                    # Ollama failed — fall back to Argos for this block
                    log_fn("    [translate] Ollama failed, falling back to Argos")
                    _translate_block_argos(block, from_code, to_code, glossary, log_fn)
            else:
                block["translated_text"] = block["text"]
        return blocks

    # --- Argos engine (default) ---
    log_fn(f"    [translate] Using Argos for {len(translatable_blocks)} block(s)")
    try:
        translate_fn = _get_translation_fn(from_code, to_code, log=log_fn)
    except Exception as exc:
        log_fn(f"    [translate] Cannot load Argos: {exc}")
        translate_fn = None

    if translate_fn is None:
        log_fn(f"    [translate] No model for {from_code}→{to_code}. Text kept as-is.")
        for block in translatable_blocks:
            block["translated_text"] = block["text"]
        return blocks

    for block in translatable_blocks:
        try:
            prefix, body = _split_bullet_prefix(block["text"])
            if body.strip():
                result = translate_fn(body)
                if result:
                    result = _post_process_translation(result, from_code, to_code)
                    if glossary:
                        result = _apply_glossary(result, glossary)
                    block["translated_text"] = prefix + result
                else:
                    block["translated_text"] = block["text"]
            else:
                block["translated_text"] = block["text"]
        except Exception as exc:
            log_fn(f"    [translate] Block failed: {exc}")
            block["translated_text"] = block["text"]

    return blocks


def _translate_block_argos(
    block: Dict[str, Any],
    from_code: str,
    to_code: str,
    glossary: Dict[str, str],
    log_fn=print,
):
    """Translate a single block using Argos (helper for Ollama fallback)."""
    try:
        translate_fn = _get_translation_fn(from_code, to_code, log=log_fn)
        if translate_fn is None:
            block["translated_text"] = block["text"]
            return
        prefix, body = _split_bullet_prefix(block["text"])
        if body.strip():
            result = translate_fn(body)
            if result:
                result = _post_process_translation(result, from_code, to_code)
                if glossary:
                    result = _apply_glossary(result, glossary)
                block["translated_text"] = prefix + result
            else:
                block["translated_text"] = block["text"]
        else:
            block["translated_text"] = block["text"]
    except Exception as exc:
        log_fn(f"    [translate] Argos fallback failed: {exc}")
        block["translated_text"] = block["text"]

"""
Language detection and translation utilities.

Language detection: langdetect  (offline, statistical)
Translation:        argostranslate (fully offline neural MT — models downloaded once,
                    then run locally with no internet connection required)
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

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

    log(f"    [translate] Translating {len(blocks)} block(s) …")
    for block in blocks:
        try:
            result = translate_fn(block["text"])
            block["translated_text"] = result if result else block["text"]
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

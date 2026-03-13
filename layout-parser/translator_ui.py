"""
Document Translation Tool — Graphical User Interface
Run with:  python translator_ui.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Self-healing launcher: if running under the wrong Python (e.g. Windows Store
# stub), restart automatically with the Python that has the packages installed.
# ---------------------------------------------------------------------------
_CORRECT_PYTHON = r"C:\Users\migne\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if os.path.exists(_CORRECT_PYTHON) and os.path.normcase(sys.executable) != os.path.normcase(_CORRECT_PYTHON):
    import subprocess
    subprocess.Popen([_CORRECT_PYTHON] + sys.argv)
    sys.exit()

import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Language lists
# ---------------------------------------------------------------------------
TARGET_LANGUAGES = [
    ("English",               "en"),
    ("French",                "fr"),
    ("German",                "de"),
    ("Spanish",               "es"),
    ("Italian",               "it"),
    ("Portuguese",            "pt"),
    ("Dutch",                 "nl"),
    ("Polish",                "pl"),
    ("Russian",               "ru"),
    ("Ukrainian",             "uk"),
    ("Turkish",               "tr"),
    ("Romanian",              "ro"),
    ("Czech",                 "cs"),
    ("Hungarian",             "hu"),
    ("Swedish",               "sv"),
    ("Norwegian",             "no"),
    ("Danish",                "da"),
    ("Finnish",               "fi"),
    ("Greek",                 "el"),
    ("Bulgarian",             "bg"),
    ("Croatian",              "hr"),
    ("Slovak",                "sk"),
    ("Lithuanian",            "lt"),
    ("Latvian",               "lv"),
    ("Estonian",              "et"),
    ("Japanese",              "ja"),
    ("Korean",                "ko"),
    ("Chinese (Simplified)",  "zh"),
    ("Chinese (Traditional)", "zt"),
    ("Arabic",                "ar"),
    ("Hebrew",                "he"),
    ("Hindi",                 "hi"),
    ("Thai",                  "th"),
    ("Vietnamese",            "vi"),
    ("Indonesian",            "id"),
    ("Malay",                 "ms"),
]

SOURCE_LANGUAGES = [("Auto-detect", "auto")] + TARGET_LANGUAGES

# Default paths
_DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _find_default_poppler() -> str:
    """Locate the poppler bin directory relative to this script."""
    base = Path(__file__).resolve().parent.parent  # project root
    for child in base.iterdir():
        if child.is_dir() and child.name.lower().startswith("poppler"):
            bin_dir = child / "Library" / "bin"
            if bin_dir.is_dir():
                return str(bin_dir)
    return ""

_DEFAULT_POPPLER = _find_default_poppler()


# ---------------------------------------------------------------------------
# Colours / styling constants
# ---------------------------------------------------------------------------
BG        = "#1e1e2e"   # dark background
PANEL     = "#2a2a3e"   # slightly lighter panels
ACCENT    = "#7c6af7"   # purple accent
ACCENT_HV = "#9d8ffa"   # hover
TEXT      = "#cdd6f4"   # main text
SUBTEXT   = "#a6adc8"   # dimmed text
SUCCESS   = "#a6e3a1"   # green
ERROR     = "#f38ba8"   # red
WARNING   = "#fab387"   # orange
ENTRY_BG  = "#313244"


class TranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Document Translator")
        self.geometry("780x700")
        self.resizable(True, True)
        self.minsize(660, 560)
        self.configure(bg=BG)

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._running = False

        self._build_ui()
        self._poll_log()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # header
        self.rowconfigure(1, weight=0)  # form
        self.rowconfigure(2, weight=1)  # log

        self._build_header()
        self._build_form()
        self._build_log()

    def _build_header(self):
        hdr = tk.Frame(self, bg=ACCENT)
        hdr.grid(row=0, column=0, sticky="ew", pady=10)
        hdr.columnconfigure(0, weight=1)
        tk.Label(
            hdr,
            text="Document Translator",
            bg=ACCENT, fg="white",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0)
        tk.Label(
            hdr,
            text="Fully offline  •  PDF, JPG, PNG  •  Powered by Tesseract + Argos Translate",
            bg=ACCENT, fg="#e0d9ff",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0)

    def _build_form(self):
        form = tk.Frame(self, bg=BG)
        form.grid(row=1, column=0, sticky="ew", padx=20, pady=16)
        form.columnconfigure(1, weight=1)

        def label(row, text):
            tk.Label(
                form, text=text, bg=BG, fg=SUBTEXT,
                font=("Segoe UI", 9), anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)

        def entry(row, textvariable, placeholder=""):
            e = tk.Entry(
                form, textvariable=textvariable,
                bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                relief="flat", font=("Segoe UI", 10),
                highlightthickness=1, highlightbackground=PANEL,
                highlightcolor=ACCENT,
            )
            e.grid(row=row, column=1, sticky="ew", pady=5)
            return e

        def browse_btn(row, command):
            return tk.Button(
                form, text="Browse…", command=command,
                bg=PANEL, fg=TEXT, relief="flat",
                font=("Segoe UI", 9), cursor="hand2",
                activebackground=ACCENT, activeforeground="white",
                padx=10,
            ).grid(row=row, column=2, padx=(8, 0), pady=5)

        # ---- Input file
        self._input_var = tk.StringVar()
        label(0, "Input document")
        entry(0, self._input_var)
        browse_btn(0, self._browse_input)

        # ---- Output folder
        self._output_var = tk.StringVar()
        label(1, "Output folder")
        entry(1, self._output_var)
        browse_btn(1, self._browse_output)

        # ---- Language row
        lang_frame = tk.Frame(form, bg=BG)
        lang_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        lang_frame.columnconfigure(1, weight=1)
        lang_frame.columnconfigure(3, weight=1)

        tk.Label(lang_frame, text="Source language", bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8))
        # Source is always auto-detected — shown as a read-only badge
        self._src_var = tk.StringVar(value="auto")
        src_badge = tk.Label(
            lang_frame, text="Auto-detect",
            bg=ENTRY_BG, fg=ACCENT,
            font=("Segoe UI", 10), padx=10, pady=4, anchor="w",
        )
        src_badge.grid(row=0, column=1, sticky="ew", padx=(0, 20))
        # hidden combobox kept for get_lang_code compatibility
        self._src_cb = ttk.Combobox(lang_frame, textvariable=self._src_var,
                                    values=["Auto-detect  (auto)"], state="readonly", width=1)
        self._src_cb.current(0)

        tk.Label(lang_frame, text="Translate to", bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=(0, 8))
        self._tgt_var = tk.StringVar(value="en")
        tgt_cb = ttk.Combobox(
            lang_frame, textvariable=self._tgt_var,
            values=[f"{name}  ({code})" for name, code in TARGET_LANGUAGES],
            state="readonly", font=("Segoe UI", 10), width=26,
        )
        tgt_cb.current(0)  # English
        tgt_cb.grid(row=0, column=3, sticky="ew")
        self._tgt_cb = tgt_cb

        # ---- Advanced (collapsible)
        self._adv_open = tk.BooleanVar(value=False)
        adv_toggle = tk.Button(
            form, text="▸ Advanced settings", bg=BG, fg=ACCENT,
            relief="flat", font=("Segoe UI", 9, "underline"),
            cursor="hand2", anchor="w", activeforeground=ACCENT_HV,
            activebackground=BG,
            command=self._toggle_advanced,
        )
        adv_toggle.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._adv_toggle_btn = adv_toggle

        self._adv_frame = tk.Frame(form, bg=BG)
        self._adv_frame.columnconfigure(1, weight=1)
        # (never gridded until toggled)

        label2 = lambda row, text: tk.Label(
            self._adv_frame, text=text, bg=BG, fg=SUBTEXT,
            font=("Segoe UI", 9), anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)

        self._tess_var = tk.StringVar(value=_DEFAULT_TESSERACT)
        label2(0, "Tesseract path")
        tk.Entry(
            self._adv_frame, textvariable=self._tess_var,
            bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Segoe UI", 9),
            highlightthickness=1, highlightbackground=PANEL,
            highlightcolor=ACCENT,
        ).grid(row=0, column=1, sticky="ew", pady=4)
        tk.Button(
            self._adv_frame, text="Browse…",
            command=lambda: self._browse_exe(self._tess_var),
            bg=PANEL, fg=TEXT, relief="flat", font=("Segoe UI", 9),
            cursor="hand2", activebackground=ACCENT, activeforeground="white",
            padx=10,
        ).grid(row=0, column=2, padx=(8, 0), pady=4)

        self._poppler_var = tk.StringVar(value=_DEFAULT_POPPLER)
        label2(1, "Poppler bin folder")
        tk.Entry(
            self._adv_frame, textvariable=self._poppler_var,
            bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Segoe UI", 9),
            highlightthickness=1, highlightbackground=PANEL,
            highlightcolor=ACCENT,
        ).grid(row=1, column=1, sticky="ew", pady=4)
        tk.Button(
            self._adv_frame, text="Browse…",
            command=lambda: self._browse_dir(self._poppler_var),
            bg=PANEL, fg=TEXT, relief="flat", font=("Segoe UI", 9),
            cursor="hand2", activebackground=ACCENT, activeforeground="white",
            padx=10,
        ).grid(row=1, column=2, padx=(8, 0), pady=4)

        self._font_var = tk.StringVar(value="")
        label2(2, "Custom font (.ttf)")
        tk.Entry(
            self._adv_frame, textvariable=self._font_var,
            bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Segoe UI", 9),
            highlightthickness=1, highlightbackground=PANEL,
            highlightcolor=ACCENT,
        ).grid(row=2, column=1, sticky="ew", pady=4)
        tk.Button(
            self._adv_frame, text="Browse…",
            command=lambda: self._browse_file(
                self._font_var, [("TrueType Font", "*.ttf *.ttc *.otf"), ("All", "*.*")]
            ),
            bg=PANEL, fg=TEXT, relief="flat", font=("Segoe UI", 9),
            cursor="hand2", activebackground=ACCENT, activeforeground="white",
            padx=10,
        ).grid(row=2, column=2, padx=(8, 0), pady=4)

        # ---- Translate button
        self._btn = tk.Button(
            form, text="Translate Document",
            command=self._start_translation,
            bg=ACCENT, fg="white",
            relief="flat", font=("Segoe UI", 11, "bold"),
            cursor="hand2", pady=10,
            activebackground=ACCENT_HV, activeforeground="white",
        )
        self._btn.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(16, 4))

        # style comboboxes
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=ENTRY_BG, background=ENTRY_BG,
                        foreground=TEXT, selectbackground=ACCENT,
                        selectforeground="white", arrowcolor=TEXT)

    def _build_log(self):
        log_frame = tk.Frame(self, bg=BG)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 16))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        tk.Label(
            log_frame, text="Progress", bg=BG, fg=SUBTEXT,
            font=("Segoe UI", 9, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self._log = tk.Text(
            log_frame, bg="#11111b", fg=TEXT,
            font=("Consolas", 9), relief="flat",
            state="disabled", wrap="word",
            highlightthickness=1, highlightbackground=PANEL,
        )
        self._log.grid(row=1, column=0, sticky="nsew")

        sb = tk.Scrollbar(log_frame, command=self._log.yview, bg=PANEL, troughcolor=BG)
        sb.grid(row=1, column=1, sticky="ns")
        self._log["yscrollcommand"] = sb.set

        self._log.tag_config("ok",   foreground=SUCCESS)
        self._log.tag_config("err",  foreground=ERROR)
        self._log.tag_config("warn", foreground=WARNING)
        self._log.tag_config("info", foreground=ACCENT)

    # ---------------------------------------------------------------- toggle --

    def _toggle_advanced(self):
        if self._adv_open.get():
            self._adv_frame.grid_forget()
            self._adv_open.set(False)
            self._adv_toggle_btn.config(text="▸ Advanced settings")
        else:
            self._adv_frame.grid(
                in_=self.nametowidget(self._btn.winfo_parent()),
                row=9, column=0, columnspan=3, sticky="ew", pady=(4, 0),
            )
            self._adv_open.set(True)
            self._adv_toggle_btn.config(text="▾ Advanced settings")

    # --------------------------------------------------------------- browse --

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select document",
            filetypes=[
                ("Supported documents", "*.pdf *.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp"),
                ("PDF files", "*.pdf"),
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._input_var.set(path)
            # Auto-fill output folder to same directory if empty
            if not self._output_var.get():
                self._output_var.set(str(Path(path).parent))

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._output_var.set(path)

    def _browse_exe(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="Select executable",
            filetypes=[("Executable", "*.exe"), ("All", "*.*")],
        )
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(title="Select folder")
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar, filetypes):
        path = filedialog.askopenfilename(title="Select file", filetypes=filetypes)
        if path:
            var.set(path)

    # ------------------------------------------------------------ translation --

    def _get_lang_code(self, cb: ttk.Combobox, all_langs) -> str:
        """Extract the ISO code from a combobox selection like 'French  (fr)'."""
        sel = cb.get()
        for name, code in all_langs:
            if sel.startswith(name):
                return code
        return sel.strip()

    def _start_translation(self):
        if self._running:
            return

        input_path = self._input_var.get().strip()
        output_dir = self._output_var.get().strip()
        src_code = self._get_lang_code(self._src_cb, SOURCE_LANGUAGES)
        tgt_code = self._get_lang_code(self._tgt_cb, TARGET_LANGUAGES)
        tess_cmd  = self._tess_var.get().strip() or None
        poppler   = self._poppler_var.get().strip() or None
        font_path = self._font_var.get().strip() or None

        if not input_path:
            messagebox.showwarning("Missing input", "Please select an input document.")
            return
        if not Path(input_path).exists():
            messagebox.showerror("File not found", f"Cannot find:\n{input_path}")
            return
        if not output_dir:
            messagebox.showwarning("Missing output", "Please select an output folder.")
            return

        # Build output path
        inp = Path(input_path)
        output_path = str(Path(output_dir) / f"{inp.stem}_translated_{tgt_code}{inp.suffix}")

        # Inject poppler into PATH for this process if provided
        if poppler and Path(poppler).exists():
            os.environ["PATH"] = poppler + os.pathsep + os.environ.get("PATH", "")

        self._log_clear()
        self._log_write(f"Input:   {input_path}\n", "info")
        self._log_write(f"Output:  {output_path}\n", "info")
        self._log_write(f"Source:  {src_code or 'auto'}  →  Target: {tgt_code}\n\n", "info")

        self._running = True
        self._btn.config(state="disabled", text="Translating…", bg="#555")

        threading.Thread(
            target=self._run_pipeline,
            args=(input_path, output_path, tgt_code, src_code, tess_cmd, font_path),
            daemon=True,
        ).start()

    def _run_pipeline(self, input_path, output_path, tgt, src, tess_cmd, font_path):
        try:
            from translator_tool.pipeline import process_document
            process_document(
                input_path=input_path,
                target_lang=tgt,
                output_path=output_path,
                source_lang=src,
                tesseract_cmd=tess_cmd,
                font_path=font_path,
                verbose=True,
                _log_fn=self._enqueue_log,
            )
            self._enqueue_log(f"\n✓ Done!  Saved to:\n  {output_path}\n", "ok")
        except Exception as exc:
            self._enqueue_log(f"\n✗ Error: {exc}\n", "err")
        finally:
            self._log_queue.put(("__done__", ""))

    def _enqueue_log(self, msg: str, tag: str = ""):
        self._log_queue.put((msg, tag))

    # --------------------------------------------------------- log helpers --

    def _log_write(self, text: str, tag: str = ""):
        self._log.config(state="normal")
        if tag:
            self._log.insert("end", text, tag)
        else:
            self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _log_clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _poll_log(self):
        """Drain the thread-safe log queue on the main thread (every 80 ms)."""
        try:
            while True:
                msg, tag = self._log_queue.get_nowait()
                if msg == "__done__":
                    self._running = False
                    self._btn.config(state="normal", text="Translate Document", bg=ACCENT)
                else:
                    self._log_write(msg, tag)
        except queue.Empty:
            pass
        self.after(80, self._poll_log)


# ---------------------------------------------------------------------------
# Patch pipeline to accept a custom log function
# ---------------------------------------------------------------------------

def _patch_pipeline():
    """Monkey-patch process_document so it accepts an optional _log_fn kwarg."""
    try:
        import translator_tool.pipeline as _pipeline

        _orig = _pipeline.process_document

        def _patched(
            input_path, target_lang, output_path=None, source_lang="auto",
            tesseract_cmd=None, font_path=None, verbose=True, _log_fn=None,
        ):
            import sys as _sys
            from pathlib import Path as _Path
            from translator_tool.file_handler import load_document, save_output, guess_output_path
            from translator_tool.pipeline import process_page

            if output_path is None:
                output_path = guess_output_path(str(input_path), target_lang)

            from translator_tool.renderer import find_system_font
            if font_path is None:
                font_path = find_system_font()

            def log(msg):
                if _log_fn:
                    # colour hints
                    if "error" in msg.lower() or "✗" in msg:
                        _log_fn(msg + "\n", "err")
                    elif "done" in msg.lower() or "✓" in msg or "saved" in msg.lower():
                        _log_fn(msg + "\n", "ok")
                    elif "warning" in msg.lower():
                        _log_fn(msg + "\n", "warn")
                    else:
                        _log_fn(msg + "\n")
                elif verbose:
                    print(msg)

            log(f"Loading document: {input_path}")
            pages = load_document(str(input_path))
            log(f"  Loaded {len(pages)} page(s).")

            processed = []
            for idx, page in enumerate(pages):
                log(f"\n── Page {idx + 1} / {len(pages)} ──────────────")
                result = process_page(
                    page_image=page,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    tesseract_cmd=tesseract_cmd,
                    font_path=font_path,
                    log=log,
                )
                processed.append(result)

            log(f"\nSaving → {output_path}")
            save_output(processed, output_path)
            return output_path

        _pipeline.process_document = _patched
    except Exception:
        pass  # if import fails, original pipeline is used


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _patch_pipeline()
    app = TranslatorApp()
    app.mainloop()

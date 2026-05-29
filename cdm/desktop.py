"""Drifter — native desktop app (PySide6 + pyqtgraph), Apple-inspired UI.

A real desktop window (no browser): guided setup connects your AI and names a goal,
then you chat in-app with streaming replies while context drift is tracked live on a
clean chart and a coach bar tells you the next step. Light/dark follow macOS.

Local by design: sessions in SQLite, API keys on disk, drift engine offline (hashing
fallback, or semantic via fastembed once downloaded). API calls go straight to the
provider. Launch with ``drifter`` (or ``python -m cdm.desktop``).
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cdm import claude_code as cc
from cdm import config
from cdm.embeddings import get_embedder
from cdm.llm import (
    PROVIDERS,
    LLMError,
    claude_cli_available,
    curated_models,
    get_key,
    key_url,
    list_models,
    load_keys,
    provider_ready,
    save_key,
)
from cdm.monitor import DriftMonitor
from cdm.transcript import parse_transcript
from cdm.watcher import is_watcher_running, start_watcher_process, stop_watcher

# --------------------------------------------------------------------------- #
# Palette (Apple-inspired light + dark) and tokenised stylesheet
# --------------------------------------------------------------------------- #
LIGHT = {
    "bg": "#FFFFFF", "panel": "#F5F5F7", "ink": "#1D1D1F", "muted": "#86868B",
    "line": "#D2D2D7", "line_soft": "#E8E8ED", "accent": "#FF6A00", "accent_hover": "#FF8124",
    "danger": "#FF3B30", "coach_bg": "#FFF2E8", "coach_fg": "#B8500E", "sel": "#FFE3CC",
    "okbg": "#E7F6EC", "okfg": "#1B7F3B", "badbg": "#FFE5E2", "badfg": "#FF3B30",
    "ubg": "#FF6A00", "ufg": "#FFFFFF", "abg": "#F5F5F7", "afg": "#1D1D1F",
    "scroll": "#C7C7CC", "input": "#FFFFFF", "hover": "#F5F5F7",
}
DARK = {
    "bg": "#1C1C1E", "panel": "#2C2C2E", "ink": "#F5F5F7", "muted": "#98989D",
    "line": "#3A3A3C", "line_soft": "#2C2C2E", "accent": "#FF7A1A", "accent_hover": "#FF9442",
    "danger": "#FF453A", "coach_bg": "#3A2A1C", "coach_fg": "#FFB37A", "sel": "#5A3A1F",
    "okbg": "#1E3A28", "okfg": "#5BD27E", "badbg": "#3A1F1E", "badfg": "#FF6B61",
    "ubg": "#FF7A1A", "ufg": "#FFFFFF", "abg": "#2C2C2E", "afg": "#F5F5F7",
    "scroll": "#48484A", "input": "#2C2C2E", "hover": "#2C2C2E",
}
C = dict(LIGHT)  # current palette (mutated by set_dark)


def set_dark(dark: bool) -> None:
    """Switch the active palette."""
    C.clear()
    C.update(DARK if dark else LIGHT)


_QSS = """
* { font-family: "-apple-system", "SF Pro Text", "SF Pro Display", "Helvetica Neue", sans-serif; }
QWidget { background: @bg@; color: @ink@; font-size: 13px; }
QMainWindow, QDialog { background: @bg@; }
QLabel#h1 { font-size: 26px; font-weight: 600; }
QLabel#h2 { font-size: 17px; font-weight: 600; }
QLabel#muted { color: @muted@; }
QLabel#anchor { color: @muted@; font-size: 12px; }
QLabel#coach { background: @coach_bg@; color: @coach_fg@; border-radius: 12px; padding: 11px 15px; font-weight: 600; }
QFrame#card { background: @bg@; border: 1px solid @line_soft@; border-radius: 16px; }
QFrame#hairline { background: @line@; max-height: 1px; min-height: 1px; border: none; }
QPushButton { background: @bg@; color: @ink@; border: 1px solid @line@; border-radius: 10px; padding: 9px 16px; font-weight: 600; }
QPushButton:hover { background: @hover@; }
QPushButton:disabled { color: @muted@; border-color: @line_soft@; }
QPushButton#primary { background: @accent@; color: #FFFFFF; border: none; }
QPushButton#primary:hover { background: @accent_hover@; }
QPushButton#primary:disabled { background: @line@; }
QPushButton#link { background: transparent; border: none; color: @accent@; padding: 9px 6px; font-weight: 600; }
QPushButton#link:hover { color: @accent_hover@; }
QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox { background: @input@; border: 1px solid @line@; border-radius: 10px; padding: 9px 11px; selection-background-color: @sel@; selection-color: @ink@; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid @accent@; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: @bg@; border: 1px solid @line@; border-radius: 10px; selection-background-color: @coach_bg@; selection-color: @ink@; outline: none; padding: 4px; }
QListWidget { background: @bg@; border: 1px solid @line_soft@; border-radius: 16px; padding: 8px; }
QListWidget::item { padding: 14px 14px; border-radius: 12px; color: @ink@; }
QListWidget::item:selected { background: @coach_bg@; color: @ink@; }
QListWidget::item:hover { background: @hover@; }
QScrollArea { border: none; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: @scroll@; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QLabel#chipOk { background: @okbg@; color: @okfg@; border-radius: 11px; padding: 6px 13px; font-weight: 700; }
QLabel#chipBad { background: @badbg@; color: @badfg@; border-radius: 11px; padding: 6px 13px; font-weight: 700; }
QLabel#bubbleUser { background: @ubg@; color: @ufg@; border-radius: 18px; padding: 11px 14px; }
QLabel#bubbleAsst { background: @abg@; color: @afg@; border-radius: 18px; padding: 11px 14px; }
QCheckBox { color: @muted@; }
"""


def build_qss() -> str:
    """Render the stylesheet against the active palette ``C``."""
    s = _QSS
    for key, value in C.items():
        s = s.replace(f"@{key}@", value)
    return s


def _md_to_html(text: str) -> str:
    """Render markdown to the HTML subset QLabel understands (with a plain fallback)."""
    try:
        from markdown_it import MarkdownIt

        return MarkdownIt("commonmark", {"breaks": True}).render(text or "")
    except Exception:
        import html as _html

        return "<span>" + _html.escape(text or "").replace("\n", "<br>") + "</span>"


def _shadow(widget: QWidget, blur: int = 30, dy: int = 8, alpha: int = 26) -> QWidget:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(eff)
    return widget


def _hairline() -> QFrame:
    f = QFrame()
    f.setObjectName("hairline")
    return f


# --------------------------------------------------------------------------- #
# Local profile + embedder preference (offline)
# --------------------------------------------------------------------------- #
def _profile_path():
    config.ensure_data_dir()
    return config.DATA_DIR / "profile.json"


def load_profile_name() -> str:
    import json

    p = _profile_path()
    if p.exists():
        try:
            return str(json.loads(p.read_text()).get("name", "") or "")
        except Exception:
            return ""
    return ""


def save_profile_name(name: str) -> None:
    import json

    _profile_path().write_text(json.dumps({"name": name}))


def time_greeting(name: str = "") -> str:
    """A time-of-day greeting (with light variety) for the chosen username."""
    hour = datetime.now().hour
    if hour < 12:
        choices = ["Good morning", "Morning", "Rise and shine"]
    elif hour < 17:
        choices = ["Good afternoon", "Afternoon"]
    elif hour < 22:
        choices = ["Good evening", "Evening"]
    else:
        choices = ["Good evening", "Working late", "Burning the midnight oil"]
    greeting = random.choice(choices)
    return f"{greeting}, {name}" if name else greeting


def first_connected_provider() -> Optional[str]:
    """First provider usable now (keyless CLI present, or a key set)."""
    return next((p for p in PROVIDERS if provider_ready(p)), None)


def default_provider() -> str:
    """Best default: a ready provider, else keyless CLI if installed, else Claude API."""
    return first_connected_provider() or ("claude-cli" if claude_cli_available() else "claude")


def safe_embedder(preference: str):
    """Build the preferred embedder, falling back to hashing on any failure."""
    try:
        return get_embedder(preference)
    except Exception:
        return get_embedder("hashing")


# --------------------------------------------------------------------------- #
# Drift chart (palette-aware area chart)
# --------------------------------------------------------------------------- #
class DriftChart(pg.PlotWidget):
    """Drift chart with a learned baseline band, changepoint marker and forecast.

    - orange line + fill: drift vs your goal
    - grey dashed: drift vs recent context
    - shaded band: the 'normal' range learned from the start of this chat
    - vertical dashed line: where a sustained shift began (changepoint)
    - dashed projection + label: forecast of when drift crosses the threshold
    - red dotted line: the alert threshold
    """

    def __init__(self) -> None:
        super().__init__()
        self.setBackground(C["bg"])
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setYRange(0, 1.0, padding=0.02)
        self.showGrid(x=False, y=True, alpha=0.10)
        for ax in ("left", "bottom"):
            self.getAxis(ax).setTextPen(C["muted"])
            self.getAxis(ax).setPen(C["line_soft"])
        self.getAxis("left").setLabel("drift", color=C["muted"])
        self.getAxis("bottom").setLabel("turn", color=C["muted"])

        mut = QColor(C["muted"])
        self._band = pg.LinearRegionItem(
            orientation="horizontal", movable=False,
            brush=pg.mkBrush(mut.red(), mut.green(), mut.blue(), 28), pen=pg.mkPen(None),
        )
        self._band.setZValue(-10)
        self.addItem(self._band)
        self._band.hide()

        accent = QColor(C["accent"])
        self._anchor = self.plot(
            [], [], pen=pg.mkPen(C["accent"], width=2.5),
            fillLevel=0, brush=pg.mkBrush(accent.red(), accent.green(), accent.blue(), 28),
            symbol="o", symbolSize=5, symbolBrush=C["accent"], symbolPen=None,
        )
        self._reference = self.plot([], [], pen=pg.mkPen(C["muted"], width=1.6, style=Qt.DashLine))
        self._forecast = self.plot([], [], pen=pg.mkPen(C["accent"], width=1.4, style=Qt.DashLine))
        self._threshold = pg.InfiniteLine(angle=0, pen=pg.mkPen(C["danger"], width=1.2, style=Qt.DotLine))
        self.addItem(self._threshold)
        self._cp = pg.InfiniteLine(angle=90, pen=pg.mkPen(C["muted"], width=1.0, style=Qt.DashLine))
        self.addItem(self._cp)
        self._cp.hide()
        self._fc_text = pg.TextItem(color=C["accent"], anchor=(0, 1))
        self.addItem(self._fc_text)
        self._fc_text.hide()

    def update(self, ts: dict) -> None:
        turns = list(ts.get("turns") or [])
        anchor = list(ts.get("drift_from_anchor") or [])
        reference = list(ts.get("drift_from_reference") or [])
        threshold = float(ts.get("threshold", 0.65))
        self._anchor.setData(turns, anchor)
        self._reference.setData(turns, reference)
        self._threshold.setValue(threshold)

        mean, std = ts.get("baseline_mean"), ts.get("baseline_std")
        if mean is not None and std is not None and len(turns) >= 2:
            self._band.setRegion([max(0.0, mean - std), mean + std])
            self._band.show()
        else:
            self._band.hide()

        cp = ts.get("changepoint_turn")
        if cp is not None:
            self._cp.setValue(cp)
            self._cp.show()
        else:
            self._cp.hide()

        fc = ts.get("forecast_turns")
        if turns and anchor and fc and fc > 0:
            x0, y0 = turns[-1], anchor[-1]
            self._forecast.setData([x0, x0 + fc], [y0, threshold])
            self._fc_text.setText(f"~{fc:.0f} turns to drift")
            self._fc_text.setPos(x0, min(1.0, threshold + 0.07))
            self._fc_text.show()
        else:
            self._forecast.setData([], [])
            self._fc_text.hide()


# --------------------------------------------------------------------------- #
# Background threads
# --------------------------------------------------------------------------- #
class ChatThread(QThread):
    """Stream one LLM reply off the UI thread. Emits chunks; supports stop()."""

    chunk = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, provider, model, api_key, messages, system) -> None:
        super().__init__()
        self._args = (provider, model, api_key, messages, system)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        provider, model, api_key, messages, system = self._args
        try:
            from cdm.llm import LLMClient

            client = LLMClient(provider, api_key=api_key, model=model)
            full = client.chat_stream(
                messages, system=system,
                on_chunk=lambda t: self.chunk.emit(t),
                should_stop=lambda: self._stop,
            )
            self.done.emit(full)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelDownloadThread(QThread):
    """Construct the semantic embedder (downloads the model once) off the UI thread."""

    done = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            get_embedder("semantic")
            self.done.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------- #
# Reusable provider/model/key setup widget
# --------------------------------------------------------------------------- #
class ProviderSetup(QWidget):
    """Provider picker + latest-model dropdown + key field with a 'Get key' link."""

    def __init__(self, provider: str = "claude", model: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(12)

        self.provider = QComboBox()
        for key, meta in PROVIDERS.items():
            self.provider.addItem(meta["label"], key)
        if provider in PROVIDERS:
            self.provider.setCurrentIndex(list(PROVIDERS).index(provider))

        self.model = QComboBox()
        self.model.setEditable(True)
        self.refresh_btn = QPushButton("Refresh list")
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(self.refresh_btn)

        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your API key")
        self.get_key_btn = QPushButton("Get an API key  ↗")
        self.get_key_btn.setObjectName("link")
        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(self.key, 1)
        key_row.addWidget(self.get_key_btn)

        self.hint = QLabel()
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)

        form.addRow("Provider", self.provider)
        form.addRow("Model", self._wrap(model_row))
        form.addRow("API key", self._wrap(key_row))
        form.addRow("", self.hint)

        self.provider.currentIndexChanged.connect(lambda _i: self._load(self._current(), None))
        self.get_key_btn.clicked.connect(self._open_key_url)
        self.refresh_btn.clicked.connect(self._refresh_models)
        self._load(self._current(), model)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _current(self) -> str:
        return self.provider.currentData()

    def _load(self, provider: str, model: Optional[str]) -> None:
        meta = PROVIDERS[provider]
        self.model.clear()
        self.model.addItems(curated_models(provider))
        self.model.setCurrentText(model or meta["default_model"])
        keyless = bool(meta.get("keyless"))
        self.key.setVisible(not keyless)
        self.get_key_btn.setVisible(not keyless)
        self.refresh_btn.setVisible(not keyless)
        if keyless:
            self.key.clear()
            if claude_cli_available():
                self.hint.setText("✓ Using your Claude subscription via Claude Code — no API key needed.")
            else:
                self.hint.setText(
                    "Claude Code isn’t installed. Install it and run `claude` once to sign "
                    "in to your subscription, or choose an API provider instead."
                )
        else:
            self.key.setText(load_keys().get(provider, ""))
            self.hint.setText(
                f"Where to find it: {meta['key_hint']}. Click ‘Get an API key’, sign in, "
                "create a key, paste it here. Stored only on this Mac."
            )

    def _open_key_url(self) -> None:
        QDesktopServices.openUrl(QUrl(key_url(self._current())))

    def _refresh_models(self) -> None:
        provider = self._current()
        key = self.key.text().strip() or get_key(provider)
        if not key:
            QMessageBox.information(self, "Drifter", "Paste your API key first, then refresh.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ids = list_models(provider, key)
        except LLMError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Drifter", str(exc))
            return
        QApplication.restoreOverrideCursor()
        current = self.model.currentText()
        self.model.clear()
        self.model.addItems(ids or curated_models(provider))
        self.model.setCurrentText(current if current in ids else (ids[0] if ids else current))

    def values(self):
        return self._current(), self.model.currentText().strip(), self.key.text().strip()

    def persist(self):
        provider, model, key = self.values()
        if key:
            save_key(provider, key)
        return provider, model, key


# --------------------------------------------------------------------------- #
# Onboarding (3 guided steps)
# --------------------------------------------------------------------------- #
class OnboardingWizard(QDialog):
    TITLES = ["Welcome", "Connect your AI", "Your goal"]

    def __init__(self, monitor: DriftMonitor, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.chosen_session_id: Optional[str] = None
        self.chosen_tail_path: Optional[str] = None
        self.provider = default_provider()
        self.model = PROVIDERS[self.provider]["default_model"]
        self.setWindowTitle("Welcome to Drifter")
        self.setMinimumSize(640, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 34, 40, 30)
        root.setSpacing(16)
        self.step_label = QLabel()
        self.step_label.setObjectName("muted")
        self.title = QLabel()
        self.title.setObjectName("h1")
        root.addWidget(self.step_label)
        root.addWidget(self.title)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._step_welcome())
        self.stack.addWidget(self._step_connect())
        self.stack.addWidget(self._step_goal())

        nav = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("primary")
        self.back_btn.clicked.connect(lambda: self._go(max(0, self.stack.currentIndex() - 1)))
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.back_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)
        self._go(0)

    def _step_welcome(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        intro = QLabel(
            "Drifter watches your AI conversation for ‘context drift’ — when the model "
            "slowly loses the thread of your goal — and helps you snap it back. "
            "Let’s set up in three quick steps."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(QLabel("What should we call you?"))
        self.name_edit = QLineEdit(load_profile_name())
        self.name_edit.setPlaceholderText("Your name (optional — stays on this Mac)")
        lay.addWidget(self.name_edit)
        lay.addStretch(1)
        return w

    def _step_connect(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        msg = QLabel(
            "Connect the AI you use. No key yet? Click ‘Get an API key’, sign in, create "
            "one, and paste it below. Calls go straight to the provider — no server of ours."
        )
        msg.setObjectName("muted")
        msg.setWordWrap(True)
        lay.addWidget(msg)
        self.setup = ProviderSetup(provider=default_provider())
        lay.addWidget(self.setup)
        lay.addStretch(1)
        return w

    def _step_goal(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)
        msg = QLabel("What’s the goal of this conversation? Drifter keeps the AI anchored to it.")
        msg.setObjectName("muted")
        msg.setWordWrap(True)
        form.addRow(msg)
        self.proj = QLineEdit()
        self.proj.setPlaceholderText("e.g. Gimbal design")
        self.goal = QPlainTextEdit()
        self.goal.setFixedHeight(90)
        self.goal.setPlaceholderText("e.g. design a pan-tilt EO/IR gimbal mount under 5 kg")
        self.cons = QPlainTextEdit()
        self.cons.setFixedHeight(70)
        self.cons.setPlaceholderText("Constraints, one per line (optional) — e.g. under 5 kg")
        form.addRow("Project", self.proj)
        form.addRow("Goal", self.goal)
        form.addRow("Constraints", self.cons)
        return w

    def _go(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.step_label.setText(f"Step {i + 1} of 3")
        self.title.setText(self.TITLES[i])
        self.back_btn.setVisible(i > 0)
        self.next_btn.setText("Finish" if i == 2 else "Next")

    def _next(self) -> None:
        i = self.stack.currentIndex()
        if i == 0:
            save_profile_name(self.name_edit.text().strip())
            self._go(1)
            return
        if i == 1:
            self.provider, self.model, _ = self.setup.persist()
            if not provider_ready(self.provider):
                ok = QMessageBox.question(
                    self, "Not connected yet",
                    "You can connect later in Settings, but chat won’t work until you do. "
                    "Continue anyway?",
                )
                if ok != QMessageBox.StandardButton.Yes:
                    return
            self._go(2)
            return
        goal = self.goal.toPlainText().strip()
        if not goal:
            QMessageBox.warning(self, "Drifter", "Please enter a goal for this session.")
            return
        cons = [c.strip() for c in self.cons.toPlainText().splitlines() if c.strip()]
        session = self.monitor.start_session(self.proj.text().strip() or "Untitled", goal, cons)
        self.chosen_session_id = session.session_id
        self.accept()


# --------------------------------------------------------------------------- #
# Settings (provider/model/key + drift-engine toggle)
# --------------------------------------------------------------------------- #
class SettingsDialog(QDialog):
    def __init__(self, provider: str, model: str, store=None, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._dl: Optional[ModelDownloadThread] = None
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(16)

        title = QLabel("Connect your AI")
        title.setObjectName("h2")
        lay.addWidget(title)
        self.setup = ProviderSetup(provider, model)
        lay.addWidget(self.setup)

        if store is not None:
            lay.addWidget(_hairline())
            eng = QLabel("Drift engine")
            eng.setObjectName("h2")
            lay.addWidget(eng)
            self.engine_state = QLabel()
            self.engine_state.setObjectName("muted")
            self.engine_state.setWordWrap(True)
            lay.addWidget(self.engine_state)
            erow = QHBoxLayout()
            fast_btn = QPushButton("Use fast (offline)")
            sem_btn = QPushButton("Enable semantic (download once)")
            fast_btn.clicked.connect(self._use_fast)
            sem_btn.clicked.connect(self._enable_semantic)
            erow.addWidget(fast_btn)
            erow.addWidget(sem_btn)
            erow.addStretch(1)
            lay.addLayout(erow)
            self._sync_engine_label()

        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        save = QPushButton("Save")
        save.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self.accept)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(save)
        lay.addLayout(row)

    def _engine_pref(self) -> str:
        return (self.store.get_meta("embedder") if self.store else None) or config.EMBEDDER_PREFERENCE

    def _sync_engine_label(self) -> None:
        pref = self._engine_pref()
        if pref in ("semantic", "fastembed"):
            self.engine_state.setText("Current: Semantic (neural, offline after download). Most accurate.")
        else:
            self.engine_state.setText(
                "Current: Fast (pure-Python, offline, zero download). Semantic is more "
                "accurate for reworded text. Changes apply on restart."
            )

    def _use_fast(self) -> None:
        if self.store:
            self.store.set_meta("embedder", "hashing")
        self._sync_engine_label()
        QMessageBox.information(self, "Drifter", "Set to Fast. Restart Drifter to apply.")

    def _enable_semantic(self) -> None:
        if self._dl and self._dl.isRunning():
            return
        prog = QProgressDialog("Downloading the semantic model (one time)…", None, 0, 0, self)
        prog.setWindowTitle("Drifter")
        prog.setWindowModality(Qt.WindowModal)
        prog.setCancelButton(None)
        prog.show()
        self._dl = ModelDownloadThread()
        self._dl.done.connect(lambda: self._semantic_done(prog, None))
        self._dl.failed.connect(lambda msg: self._semantic_done(prog, msg))
        self._dl.start()

    def _semantic_done(self, prog, error: Optional[str]) -> None:
        prog.close()
        if error:
            QMessageBox.warning(self, "Drifter", f"Could not enable semantic: {error}")
            return
        if self.store:
            self.store.set_meta("embedder", "semantic")
        self._sync_engine_label()
        QMessageBox.information(self, "Drifter", "Semantic drift enabled. Restart Drifter to apply.")

    def result_values(self):
        return self.setup.persist()


class NewSessionDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New session")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(14)
        form = QFormLayout()
        form.setSpacing(12)
        self.name = QLineEdit()
        self.goal = QPlainTextEdit()
        self.goal.setFixedHeight(84)
        self.goal.setPlaceholderText("The north-star goal this conversation should stay on")
        self.constraints = QPlainTextEdit()
        self.constraints.setFixedHeight(64)
        self.constraints.setPlaceholderText("One per line (optional)")
        form.addRow("Project", self.name)
        form.addRow("Goal", self.goal)
        form.addRow("Constraints", self.constraints)
        lay.addLayout(form)
        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        create = QPushButton("Create")
        create.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        create.clicked.connect(self.accept)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(create)
        lay.addLayout(row)

    def values(self):
        cons = [c.strip() for c in self.constraints.toPlainText().splitlines() if c.strip()]
        return self.name.text().strip(), self.goal.toPlainText().strip(), cons


def start_cc_monitoring(monitor: DriftMonitor, cc_path: str) -> Optional[str]:
    """Create a Drifter session from a Claude Code transcript (anchor = first
    prompt; ingest the rest). Returns the new session id, or None if empty."""
    turns = cc.parse_transcript_file(cc_path)
    if not turns:
        return None
    anchor = turns[0]["text"]
    label = (anchor[:40] + "…") if len(anchor) > 40 else anchor
    session = monitor.start_session(f"Claude Code · {label}", anchor, [])
    rest = [{"role": t["role"], "text": t["text"]} for t in turns[1:]]
    if rest:
        monitor.ingest_transcript(session.session_id, rest)
    return session.session_id


class ClaudeCodeDialog(QDialog):
    """Pick a Claude Code session to monitor (live-tail its transcript)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.chosen_path: Optional[str] = None
        self.setWindowTitle("Monitor a Claude Code session")
        self.setMinimumSize(640, 540)
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(12)
        title = QLabel("Monitor your Claude Code terminal")
        title.setObjectName("h1")
        root.addWidget(title)
        sub = QLabel(
            "Pick a session. Drifter reads its transcript and draws the drift graph "
            "live while you keep chatting in your `claude` terminal — nothing to paste."
        )
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self.list = QListWidget()
        self._sessions = cc.list_sessions()
        for s in self._sessions:
            item = QListWidgetItem(f"{s['title']}\n{s['cwd']}  ·  {s['session_id'][:8]}")
            item.setData(Qt.UserRole, s["path"])
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _i: self._pick())
        root.addWidget(self.list, 1)
        if not self._sessions:
            empty = QLabel("No Claude Code sessions found under ~/.claude/projects.")
            empty.setObjectName("muted")
            root.addWidget(empty)

        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        refresh = QPushButton("Refresh")
        pick = QPushButton("Monitor")
        pick.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        refresh.clicked.connect(self._refresh)
        pick.clicked.connect(self._pick)
        row.addWidget(cancel)
        row.addWidget(refresh)
        row.addStretch(1)
        row.addWidget(pick)
        root.addLayout(row)

    def _refresh(self) -> None:
        self.list.clear()
        self._sessions = cc.list_sessions()
        for s in self._sessions:
            item = QListWidgetItem(f"{s['title']}\n{s['cwd']}  ·  {s['session_id'][:8]}")
            item.setData(Qt.UserRole, s["path"])
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _pick(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self.chosen_path = item.data(Qt.UserRole)
        self.accept()


class LaunchDialog(QDialog):
    def __init__(self, monitor: DriftMonitor, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.chosen_session_id: Optional[str] = None
        self.chosen_tail_path: Optional[str] = None
        self.setWindowTitle("Drifter")
        self.setMinimumSize(580, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 28)
        root.setSpacing(14)
        name = load_profile_name()
        hello = QLabel(time_greeting(name))
        hello.setObjectName("h1")
        root.addWidget(hello)
        prompt = QLabel("Which session do you want to continue today?")
        prompt.setObjectName("muted")
        root.addWidget(prompt)

        self.list = QListWidget()
        self._populate()
        self.list.itemDoubleClicked.connect(lambda _i: self._continue())
        root.addWidget(self.list, 1)

        manage = QHBoxLayout()
        rename_btn = QPushButton("Rename")
        delete_btn = QPushButton("Delete")
        cc_btn = QPushButton("Monitor Claude Code…")
        rename_btn.clicked.connect(self._rename)
        delete_btn.clicked.connect(self._delete)
        cc_btn.clicked.connect(self._monitor_cc)
        manage.addWidget(rename_btn)
        manage.addWidget(delete_btn)
        manage.addStretch(1)
        manage.addWidget(cc_btn)
        root.addLayout(manage)

        btns = QHBoxLayout()
        new_btn = QPushButton("New session")
        imp_btn = QPushButton("Import chat…")
        conn_btn = QPushButton("Settings")
        cont_btn = QPushButton("Continue")
        cont_btn.setObjectName("primary")
        new_btn.clicked.connect(self._new)
        imp_btn.clicked.connect(self._import)
        conn_btn.clicked.connect(self._connect)
        cont_btn.clicked.connect(self._continue)
        btns.addWidget(new_btn)
        btns.addWidget(imp_btn)
        btns.addWidget(conn_btn)
        btns.addStretch(1)
        btns.addWidget(cont_btn)
        root.addLayout(btns)

    def _populate(self) -> None:
        self.list.clear()
        for s in self.monitor.store.list_sessions():
            try:
                n = len(self.monitor.store.get_messages(s.session_id))
            except Exception:
                n = 0
            item = QListWidgetItem(
                f"{s.project_name}\n{n} turns · {s.updated_at[:16].replace('T', '  ')}"
            )
            item.setData(Qt.UserRole, s.session_id)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _selected_id(self) -> Optional[str]:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _new(self) -> None:
        dlg = NewSessionDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name, goal, cons = dlg.values()
            if not goal:
                QMessageBox.warning(self, "Drifter", "A goal is required.")
                return
            session = self.monitor.start_session(name or "Untitled", goal, cons)
            self.chosen_session_id = session.session_id
            self.accept()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a chat transcript", "", "Transcripts (*.json *.md *.txt);;All files (*)"
        )
        if not path:
            return
        turns = parse_transcript(path)
        if not turns:
            QMessageBox.warning(self, "Drifter", "No turns could be parsed from that file.")
            return
        session = self.monitor.start_session("Imported chat", turns[0]["text"], [])
        if len(turns) > 1:
            self.monitor.ingest_transcript(session.session_id, turns[1:])
        self.chosen_session_id = session.session_id
        self.accept()

    def _rename(self) -> None:
        sid = self._selected_id()
        if not sid:
            return
        session = self.monitor.store.get_session(sid)
        if session is None:
            return
        name, ok = QInputDialog.getText(self, "Rename session", "Project name:", text=session.project_name)
        if ok and name.strip():
            session.project_name = name.strip()
            self.monitor.store.update_session(session)
            self._populate()

    def _delete(self) -> None:
        sid = self._selected_id()
        if not sid:
            return
        if QMessageBox.question(self, "Delete session", "Delete this session permanently?") \
                == QMessageBox.StandardButton.Yes:
            self.monitor.store.delete_session(sid)
            self._populate()

    def _monitor_cc(self) -> None:
        dlg = ClaudeCodeDialog(self)
        if dlg.exec() != QDialog.Accepted or not dlg.chosen_path:
            return
        try:
            sid = start_cc_monitoring(self.monitor, dlg.chosen_path)
        except Exception as exc:
            QMessageBox.warning(self, "Drifter", f"Could not read that session: {exc}")
            return
        if not sid:
            QMessageBox.warning(self, "Drifter", "That session has no messages yet.")
            return
        self.chosen_session_id = sid
        self.chosen_tail_path = dlg.chosen_path
        self.accept()

    def _connect(self) -> None:
        provider = first_connected_provider() or "claude"
        SettingsDialog(provider, PROVIDERS[provider]["default_model"], self.monitor.store, self).exec()

    def _continue(self) -> None:
        sid = self._selected_id()
        if not sid:
            QMessageBox.information(self, "Drifter", "Pick a session, or create/import one.")
            return
        self.chosen_session_id = sid
        self.accept()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, monitor: DriftMonitor, session_id: str, tail_path: Optional[str] = None) -> None:
        super().__init__()
        self.monitor = monitor
        self.session_id = session_id
        self.provider = default_provider()
        self.model = PROVIDERS[self.provider]["default_model"]
        self._thread: Optional[ChatThread] = None
        self._stream_label: Optional[QLabel] = None
        self._stream_text = ""
        self._bubbles: List[QWidget] = []
        # Tail mode: monitor a live Claude Code terminal transcript (read-only).
        self.tail = cc.ClaudeCodeTail(tail_path, start_at_end=True) if tail_path else None

        session = monitor.store.get_session(session_id)
        self.setWindowTitle(f"Drifter — {session.project_name if session else ''}")
        self.resize(1180, 768)
        self._build_ui(session)
        self._refresh_chart()
        self._update_coach()
        self._update_buttons()

        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -- layout -------------------------------------------------------------- #
    def _build_ui(self, session) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(14)

        head = QHBoxLayout()
        title = QLabel(session.project_name if session else "Drifter")
        title.setObjectName("h1")
        head.addWidget(title)
        head.addStretch(1)
        self.provider_label = QLabel()
        self.provider_label.setObjectName("muted")
        head.addWidget(self.provider_label)
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        head.addWidget(settings_btn)
        outer.addLayout(head)

        anchor = QLabel(f"Anchor goal — {session.anchor_goal if session else ''}")
        anchor.setObjectName("anchor")
        anchor.setWordWrap(True)
        outer.addWidget(anchor)

        self.coach = QLabel("")
        self.coach.setObjectName("coach")
        self.coach.setWordWrap(True)
        outer.addWidget(self.coach)
        outer.addWidget(_hairline())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_chat_panel())
        split.addWidget(self._build_drift_panel())
        split.setSizes([590, 560])
        outer.addWidget(split, 1)
        self._sync_provider_label()

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(10)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_inner = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_inner)
        self.chat_layout.setContentsMargins(2, 2, 8, 2)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_inner)
        lay.addWidget(self.chat_scroll, 1)

        for m in self.monitor.store.get_messages(self.session_id):
            self._add_bubble(m.role, m.text, rich=(m.role or "").lower() != "user")

        self.status = QLabel("")
        self.status.setObjectName("muted")
        lay.addWidget(self.status)

        self.input = QPlainTextEdit()
        self.input.setFixedHeight(76)
        self.input.setPlaceholderText("Message your AI…   (⌘/Ctrl + Enter to send)")
        lay.addWidget(self.input)

        brow = QHBoxLayout()
        self.regen_btn = QPushButton("Regenerate")
        self.regen_btn.clicked.connect(self._on_regenerate)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self._on_send)
        brow.addWidget(self.regen_btn)
        brow.addStretch(1)
        brow.addWidget(self.stop_btn)
        brow.addWidget(self.send_btn)
        lay.addLayout(brow)

        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self._on_send)
        QShortcut(QKeySequence("Meta+Return"), self.input, activated=self._on_send)

        if self.tail:  # read-only monitor: chatting happens in the terminal
            self.input.hide()
            self.regen_btn.hide()
            self.stop_btn.hide()
            self.send_btn.hide()
            self.status.setText("● Monitoring your Claude Code terminal — chat there; this graph updates live.")
        return panel

    def _build_drift_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(12)

        top = QHBoxLayout()
        self.chip = QLabel("on track")
        self.chip.setObjectName("chipOk")
        self.metric = QLabel("drift 0.000")
        self.metric.setObjectName("muted")
        top.addWidget(self.chip)
        top.addStretch(1)
        top.addWidget(self.metric)
        lay.addLayout(top)

        chart_card = QFrame()
        chart_card.setObjectName("card")
        cc = QVBoxLayout(chart_card)
        cc.setContentsMargins(12, 12, 12, 12)
        self.chart = DriftChart()
        self.chart.setMinimumHeight(250)
        cc.addWidget(self.chart)
        _shadow(chart_card)
        lay.addWidget(chart_card, 1)

        th_row = QHBoxLayout()
        th_lbl = QLabel("Threshold")
        th_lbl.setObjectName("muted")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.30, 0.95)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(round(self.monitor.threshold, 2))
        self.threshold_spin.setToolTip(
            "How far the conversation may drift from your goal before Drifter warns you.\n"
            "Lower = stricter. Drift: 0 = on goal, 1 = unrelated."
        )
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        help_btn = QPushButton("?")
        help_btn.setObjectName("link")
        help_btn.setFixedWidth(26)
        help_btn.setToolTip("What does the threshold mean?")
        help_btn.clicked.connect(self._explain_threshold)
        self.auto_check = QCheckBox("Auto re-align")
        self.auto_check.setChecked(True)
        self.clip_check = QCheckBox("Capture clipboard")
        self.clip_check.setChecked(is_watcher_running())
        self.clip_check.toggled.connect(self._on_clip_toggle)
        th_row.addWidget(th_lbl)
        th_row.addWidget(self.threshold_spin)
        th_row.addWidget(help_btn)
        th_row.addStretch(1)
        th_row.addWidget(self.auto_check)
        th_row.addWidget(self.clip_check)
        lay.addLayout(th_row)

        legend = QLabel(
            f"<span style='color:{C['accent']}'>●</span> vs your goal &nbsp;&nbsp;"
            f"<span style='color:{C['muted']}'>●</span> vs recent context &nbsp;&nbsp;"
            f"<span style='color:{C['danger']}'>●</span> alert threshold &nbsp;&nbsp;"
            f"<span style='color:{C['muted']}'>▭</span> normal range"
        )
        legend.setTextFormat(Qt.RichText)
        lay.addWidget(legend)
        explain = QLabel(
            "Drift 0 = right on your goal · 1 = unrelated. Above the red line = off-track; "
            "the shaded band is what’s normal for this chat, so rising above it signals a "
            "real shift, not noise."
        )
        explain.setObjectName("muted")
        explain.setWordWrap(True)
        lay.addWidget(explain)

        self.corr_card = QFrame()
        self.corr_card.setObjectName("card")
        ccc = QVBoxLayout(self.corr_card)
        ccc.setContentsMargins(16, 14, 16, 14)
        ct = QLabel("Drift detected — re-align")
        ct.setObjectName("h2")
        ccc.addWidget(ct)
        self.corr_text = QPlainTextEdit()
        self.corr_text.setReadOnly(True)
        self.corr_text.setFixedHeight(118)
        ccc.addWidget(self.corr_text)
        crow = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        send_corr = QPushButton("Send to re-align")
        send_corr.setObjectName("primary")
        copy_btn.clicked.connect(self._copy_corrective)
        send_corr.clicked.connect(self._send_corrective)
        crow.addStretch(1)
        crow.addWidget(copy_btn)
        crow.addWidget(send_corr)
        ccc.addLayout(crow)
        _shadow(self.corr_card)
        lay.addWidget(self.corr_card)
        self.corr_card.setVisible(False)
        return panel

    # -- chat bubbles -------------------------------------------------------- #
    def _add_bubble(self, role: str, text: str, rich: bool = False) -> QLabel:
        is_user = (role or "").lower() == "user"
        bubble = QLabel()
        bubble.setObjectName("bubbleUser" if is_user else "bubbleAsst")
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(470)
        if rich:
            bubble.setTextFormat(Qt.RichText)
            bubble.setText(_md_to_html(text))
        else:
            bubble.setTextFormat(Qt.PlainText)
            bubble.setText(text)
        wrap = QHBoxLayout()
        if is_user:
            wrap.addStretch(1)
            wrap.addWidget(bubble)
        else:
            wrap.addWidget(bubble)
            wrap.addStretch(1)
        container = QWidget()
        container.setLayout(wrap)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, container)
        self._bubbles.append(container)
        QTimer.singleShot(30, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()))
        return bubble

    def _pop_last_bubble(self) -> None:
        if not self._bubbles:
            return
        container = self._bubbles.pop()
        self.chat_layout.removeWidget(container)
        container.deleteLater()

    # -- coach + labels ------------------------------------------------------ #
    def _sync_provider_label(self) -> None:
        dot = "●" if provider_ready(self.provider) else "○"
        self.provider_label.setText(f"{dot} {PROVIDERS[self.provider]['label']} · {self.model}")

    def _update_coach(self) -> None:
        if self.tail:
            if self._last_drift_high():
                self.coach.setText(
                    "③ Drift detected in your Claude Code chat — see the corrective prompt "
                    "on the right; paste it into your terminal to re-align."
                )
            else:
                self.coach.setText(
                    "● Monitoring your Claude Code terminal — keep chatting there; the "
                    "graph updates live."
                )
            return
        n = len(self.monitor.store.get_messages(self.session_id))
        if not provider_ready(self.provider):
            self.coach.setText(
                "① Connect your AI to start — open Settings (top right). Use your Claude "
                "subscription (no key) if you have Claude Code, or paste an API key."
            )
        elif n == 0:
            self.coach.setText("② You’re connected. Type your first message below and press Send.")
        elif self._last_drift_high():
            self.coach.setText("③ Drift detected — review the corrective prompt and ‘Send to re-align’.")
        else:
            self.coach.setText("Monitoring — keep chatting. The chart updates live as drift changes.")

    def _last_drift_high(self) -> bool:
        ts = self.monitor.timeseries(self.session_id)
        anchor = ts.get("drift_from_anchor") or []
        return bool(anchor) and anchor[-1] > float(ts.get("threshold", self.monitor.threshold))

    def _busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.input.setReadOnly(busy)
        self.stop_btn.setEnabled(busy)
        self.status.setText("Streaming…" if busy else "")
        if not busy:
            self._update_buttons()

    def _update_buttons(self) -> None:
        msgs = self.monitor.store.get_messages(self.session_id)
        can_regen = bool(msgs) and (msgs[-1].role or "").lower() == "assistant"
        running = bool(self._thread and self._thread.isRunning())
        self.regen_btn.setEnabled(can_regen and not running)
        self.stop_btn.setEnabled(running)

    # -- actions ------------------------------------------------------------- #
    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.provider, self.model, self.monitor.store, self)
        if dlg.exec() == QDialog.Accepted:
            provider, model, _ = dlg.result_values()
            self.provider = provider
            self.model = model or PROVIDERS[provider]["default_model"]
            self._sync_provider_label()
            self._update_coach()

    def _explain_threshold(self) -> None:
        QMessageBox.information(
            self, "What the threshold means",
            "Drift is how far the conversation has moved from your original goal — "
            "0 means right on it, 1 means completely unrelated.\n\n"
            "The threshold (red line) is how much drift you'll tolerate before Drifter "
            "warns you and offers a corrective prompt. Lower = stricter.\n\n"
            "Defaults: 0.65 in semantic mode, 0.80 in fast offline mode (its numbers run "
            "higher).\n\n"
            "The shaded band is the ‘normal’ range learned from the start of THIS "
            "conversation, so a rise above the band is a genuine shift rather than noise. "
            "The dashed projection forecasts when drift will cross the threshold.",
        )

    def _on_threshold(self, value: float) -> None:
        self.monitor.set_threshold(float(value))
        self._refresh_chart()

    def _on_clip_toggle(self, on: bool) -> None:
        if on:
            self.monitor.store.set_active_session(self.session_id)
            start_watcher_process(db_path=self.monitor.store.db_path)
        else:
            stop_watcher()

    def _system_prompt(self, realign: bool) -> str:
        session = self.monitor.store.get_session(self.session_id)
        base = f"You are helping the user with this goal: {session.anchor_goal}." if session else ""
        if session and session.constraints:
            base += " Constraints: " + "; ".join(session.constraints) + "."
        if realign:
            base += "\n\n" + self.monitor.current_corrective_prompt(self.session_id)
        return base.strip()

    def _history(self) -> List[dict]:
        return [
            {"role": m.role, "content": m.text}
            for m in self.monitor.store.get_messages(self.session_id)
        ]

    def _start_stream(self, realign: bool) -> None:
        """Open a live assistant bubble and stream a reply into it."""
        self._stream_text = ""
        self._stream_label = self._add_bubble("assistant", "▍", rich=False)
        self._busy(True)
        self._thread = ChatThread(
            self.provider, self.model, get_key(self.provider), self._history(),
            self._system_prompt(realign),
        )
        self._thread.chunk.connect(self._on_chunk)
        self._thread.done.connect(self._on_stream_done)
        self._thread.failed.connect(self._on_reply_error)
        self._thread.start()

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or (self._thread and self._thread.isRunning()):
            return
        if not provider_ready(self.provider):
            QMessageBox.information(self, "Drifter", "Connect your AI in Settings first.")
            self._open_settings()
            return
        self.input.clear()
        res = self.monitor.add_turn(self.session_id, "user", text)
        self._add_bubble("user", text)
        self._refresh_chart()
        self._start_stream(bool(self.auto_check.isChecked() and res.get("alert")))

    def _on_chunk(self, delta: str) -> None:
        if self._stream_label is None:
            return
        self._stream_text += delta
        self._stream_label.setText(self._stream_text + "▍")
        self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())

    def _on_stream_done(self, full: str) -> None:
        full = full or self._stream_text
        if self._stream_label is not None:
            self._stream_label.setTextFormat(Qt.RichText)
            self._stream_label.setText(_md_to_html(full) if full else "<i>(no response)</i>")
        self._stream_label = None
        self._busy(False)
        if full:
            self.monitor.add_turn(self.session_id, "assistant", full)
        self._refresh_chart()
        self._update_coach()
        self._update_buttons()

    def _on_reply_error(self, message: str) -> None:
        if self._stream_label is not None:
            self._pop_last_bubble()
            self._stream_label = None
        self._busy(False)
        QMessageBox.warning(self, "LLM error", message)

    def _on_stop(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self.status.setText("Stopping…")

    def _on_regenerate(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        msgs = self.monitor.store.get_messages(self.session_id)
        if not msgs or (msgs[-1].role or "").lower() != "assistant":
            return
        self.monitor.remove_last_turn(self.session_id)
        self._pop_last_bubble()
        self._refresh_chart()
        self._start_stream(False)

    def _copy_corrective(self) -> None:
        QApplication.clipboard().setText(self.corr_text.toPlainText())
        self.status.setText("Corrective prompt copied.")

    def _send_corrective(self) -> None:
        if (self._thread and self._thread.isRunning()):
            return
        if not provider_ready(self.provider):
            self._open_settings()
            return
        prompt = self.monitor.current_corrective_prompt(self.session_id)
        self.monitor.add_turn(self.session_id, "user", prompt)
        self._add_bubble("user", "↻ Re-align: corrective prompt sent")
        self._refresh_chart()
        self._start_stream(False)

    # -- refresh ------------------------------------------------------------- #
    def _tick(self) -> None:
        if self.tail:
            try:
                for t in self.tail.new_turns():
                    self.monitor.add_turn(self.session_id, t["role"], t["text"])
                    self._add_bubble(t["role"], t["text"], rich=(t["role"] != "user"))
            except Exception:
                pass
        self._refresh_chart()
        self._update_coach()
        self._update_buttons()

    def _refresh_chart(self) -> None:
        try:
            ts = self.monitor.timeseries(self.session_id)
        except Exception:
            return
        turns = ts.get("turns") or []
        anchor = ts.get("drift_from_anchor") or []
        reference = ts.get("drift_from_reference") or []
        threshold = float(ts.get("threshold", self.monitor.threshold))
        self.chart.update(ts)
        last = anchor[-1] if anchor else 0.0
        high = bool(turns) and (last > threshold)
        self.metric.setText(f"drift {last:.3f} · {len(turns)} turns")
        if high:
            self.chip.setText("DRIFTING")
            self.chip.setObjectName("chipBad")
            self.corr_text.setPlainText(self.monitor.current_corrective_prompt(self.session_id))
            self.corr_card.setVisible(True)
        else:
            self.chip.setText("on track")
            self.chip.setObjectName("chipOk")
            self.corr_card.setVisible(False)
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            if self._thread and self._thread.isRunning():
                self._thread.stop()
                self._thread.wait(2000)
            if is_watcher_running():
                stop_watcher()
        except Exception:
            pass
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    """Launch the Drifter desktop app."""
    pg.setConfigOptions(antialias=True)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Drifter")

    try:
        set_dark(app.styleHints().colorScheme() == Qt.ColorScheme.Dark)
    except Exception:
        set_dark(False)
    app.setStyleSheet(build_qss())
    try:
        app.styleHints().colorSchemeChanged.connect(
            lambda _s: (set_dark(app.styleHints().colorScheme() == Qt.ColorScheme.Dark),
                        app.setStyleSheet(build_qss()))
        )
    except Exception:
        pass
    app.setFont(QFont("-apple-system", 13))

    from cdm.storage import Store

    store = Store()
    pref = store.get_meta("embedder") or config.EMBEDDER_PREFERENCE
    monitor = DriftMonitor(store=store, embedder=safe_embedder(pref))

    first_run = not load_profile_name() and not store.list_sessions()
    if first_run:
        dlg: QDialog = OnboardingWizard(monitor)
    else:
        dlg = LaunchDialog(monitor)
    if dlg.exec() != QDialog.Accepted or not dlg.chosen_session_id:
        return 0

    monitor.store.set_active_session(dlg.chosen_session_id)
    window = MainWindow(monitor, dlg.chosen_session_id, tail_path=getattr(dlg, "chosen_tail_path", None))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

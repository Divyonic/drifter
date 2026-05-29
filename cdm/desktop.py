"""Drifter — native desktop app (PySide6 + pyqtgraph), Apple-inspired UI.

A real desktop window (no browser): a guided setup walks you through connecting
your AI and naming a goal, then you chat in-app while context drift is tracked live
on a clean chart and a coach bar tells you the next step. Everything is local —
sessions in SQLite, API keys on disk, drift engine offline; API calls go straight
to the provider.

Launch with ``drifter`` (or ``python -m cdm.desktop``).
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
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cdm import config
from cdm.llm import (
    PROVIDERS,
    LLMError,
    curated_models,
    get_key,
    key_url,
    list_models,
    load_keys,
    save_key,
)
from cdm.monitor import DriftMonitor
from cdm.transcript import parse_transcript
from cdm.watcher import is_watcher_running, start_watcher_process, stop_watcher

# --------------------------------------------------------------------------- #
# Palette (Apple-inspired: white, system grays, orange accent) + theme
# --------------------------------------------------------------------------- #
C = {
    "bg": "#FFFFFF",
    "panel": "#F5F5F7",
    "ink": "#1D1D1F",
    "muted": "#86868B",
    "line": "#D2D2D7",
    "line_soft": "#E8E8ED",
    "accent": "#FF6A00",
    "accent_hover": "#FF8124",
    "danger": "#FF3B30",
}

QSS = """
* { font-family: "-apple-system", "SF Pro Text", "SF Pro Display", "Helvetica Neue", sans-serif; }
QWidget { background: #FFFFFF; color: #1D1D1F; font-size: 13px; }
QMainWindow, QDialog { background: #FFFFFF; }
QLabel#h1 { font-size: 26px; font-weight: 600; }
QLabel#h2 { font-size: 17px; font-weight: 600; }
QLabel#muted { color: #86868B; }
QLabel#anchor { color: #86868B; font-size: 12px; }
QLabel#coach { background: #FFF2E8; color: #B8500E; border-radius: 12px; padding: 11px 15px; font-weight: 600; }

QFrame#card { background: #FFFFFF; border: 1px solid #E8E8ED; border-radius: 16px; }
QFrame#hairline { background: #D2D2D7; max-height: 1px; min-height: 1px; border: none; }

QPushButton { background: #FFFFFF; color: #1D1D1F; border: 1px solid #D2D2D7; border-radius: 10px; padding: 9px 16px; font-weight: 600; }
QPushButton:hover { background: #F5F5F7; }
QPushButton:disabled { color: #B0B0B5; border-color: #E8E8ED; }
QPushButton#primary { background: #FF6A00; color: #FFFFFF; border: none; }
QPushButton#primary:hover { background: #FF8124; }
QPushButton#primary:disabled { background: #FFCBA6; }
QPushButton#link { background: transparent; border: none; color: #FF6A00; padding: 9px 6px; font-weight: 600; }
QPushButton#link:hover { color: #FF8124; }

QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox {
    background: #FFFFFF; border: 1px solid #D2D2D7; border-radius: 10px; padding: 9px 11px;
    selection-background-color: #FFE3CC; selection-color: #1D1D1F;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid #FF6A00; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #FFFFFF; border: 1px solid #D2D2D7; border-radius: 10px;
    selection-background-color: #FFF2E8; selection-color: #1D1D1F; outline: none; padding: 4px; }

QListWidget { background: #FFFFFF; border: 1px solid #E8E8ED; border-radius: 16px; padding: 8px; }
QListWidget::item { padding: 14px 14px; border-radius: 12px; color: #1D1D1F; }
QListWidget::item:selected { background: #FFF2E8; color: #1D1D1F; }
QListWidget::item:hover { background: #F5F5F7; }

QScrollArea { border: none; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: #C7C7CC; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }

QLabel#chipOk { background: #E7F6EC; color: #1B7F3B; border-radius: 11px; padding: 6px 13px; font-weight: 700; }
QLabel#chipBad { background: #FFE5E2; color: #FF3B30; border-radius: 11px; padding: 6px 13px; font-weight: 700; }
QLabel#bubbleUser { background: #FF6A00; color: #FFFFFF; border-radius: 18px; padding: 11px 14px; }
QLabel#bubbleAsst { background: #F5F5F7; color: #1D1D1F; border-radius: 18px; padding: 11px 14px; }
QCheckBox { color: #86868B; }
"""


def _shadow(widget: QWidget, blur: int = 30, dy: int = 8, alpha: int = 26) -> QWidget:
    """Apply a soft Apple-like drop shadow to a widget (usually a card)."""
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
# Local profile (offline; just a display name)
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
    """Return the first provider that already has a key, if any."""
    return next((p for p in PROVIDERS if get_key(p)), None)


def any_key_present() -> bool:
    return first_connected_provider() is not None


# --------------------------------------------------------------------------- #
# Drift chart (clean area chart)
# --------------------------------------------------------------------------- #
class DriftChart(pg.PlotWidget):
    """Minimal area chart: orange = drift vs anchor, grey dashed = vs reference."""

    def __init__(self) -> None:
        super().__init__()
        self.setBackground(C["bg"])
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setYRange(0, 1.0, padding=0.02)
        self.showGrid(x=False, y=True, alpha=0.08)
        for ax in ("left", "bottom"):
            self.getAxis(ax).setTextPen(C["muted"])
            self.getAxis(ax).setPen(C["line_soft"])
        self.getAxis("left").setLabel("drift", color=C["muted"])
        self.getAxis("bottom").setLabel("turn", color=C["muted"])

        self._anchor = self.plot(
            [], [], pen=pg.mkPen(C["accent"], width=2.5),
            fillLevel=0, brush=pg.mkBrush(255, 106, 0, 28),
            symbol="o", symbolSize=5, symbolBrush=C["accent"], symbolPen=None,
        )
        self._reference = self.plot(
            [], [], pen=pg.mkPen("#C7C7CC", width=1.6, style=Qt.DashLine)
        )
        self._threshold = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(C["danger"], width=1.2, style=Qt.DotLine)
        )
        self.addItem(self._threshold)

    def update_series(self, turns, anchor, reference, threshold: float) -> None:
        self._anchor.setData(list(turns), list(anchor))
        self._reference.setData(list(turns), list(reference))
        self._threshold.setValue(float(threshold))


# --------------------------------------------------------------------------- #
# Background LLM call
# --------------------------------------------------------------------------- #
class ChatThread(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, provider, model, api_key, messages, system) -> None:
        super().__init__()
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._messages = messages
        self._system = system

    def run(self) -> None:
        try:
            from cdm.llm import LLMClient

            client = LLMClient(self._provider, api_key=self._api_key, model=self._model)
            self.done.emit(client.chat(self._messages, system=self._system))
        except Exception as exc:
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------- #
# Reusable provider/model/key setup widget (used by onboarding + settings)
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
# Guided onboarding (3 steps)
# --------------------------------------------------------------------------- #
class OnboardingWizard(QDialog):
    """First-run setup: welcome → connect AI → goal. Produces a session id."""

    TITLES = ["Welcome", "Connect your AI", "Your goal"]

    def __init__(self, monitor: DriftMonitor, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.chosen_session_id: Optional[str] = None
        self.provider = "claude"
        self.model = PROVIDERS["claude"]["default_model"]
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
            "Connect the AI you use. Don’t have a key yet? Click ‘Get an API key’, sign "
            "in, create one, and paste it below. Calls go straight to the provider — "
            "there’s no server of ours."
        )
        msg.setObjectName("muted")
        msg.setWordWrap(True)
        lay.addWidget(msg)
        self.setup = ProviderSetup()
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
            if not get_key(self.provider):
                ok = QMessageBox.question(
                    self, "No key yet",
                    "You can add a key later in Settings, but chat won’t work until you "
                    "do. Continue anyway?",
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
# Settings + new-session + launch dialogs
# --------------------------------------------------------------------------- #
class SettingsDialog(QDialog):
    """Change the active provider, model and (locally stored) API key."""

    def __init__(self, provider: str, model: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect your AI")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(16)
        title = QLabel("Connect your AI")
        title.setObjectName("h2")
        lay.addWidget(title)
        self.setup = ProviderSetup(provider, model)
        lay.addWidget(self.setup)
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

    def result_values(self):
        return self.setup.persist()


class NewSessionDialog(QDialog):
    """Collect project, goal and constraints for a new session."""

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


class LaunchDialog(QDialog):
    """Returning-user startup: greet, list past sessions, continue/new/import."""

    def __init__(self, monitor: DriftMonitor, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.chosen_session_id: Optional[str] = None
        self.setWindowTitle("Drifter")
        self.setMinimumSize(580, 580)

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

        btns = QHBoxLayout()
        new_btn = QPushButton("New session")
        imp_btn = QPushButton("Import chat…")
        conn_btn = QPushButton("Connect AI")
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

    def _connect(self) -> None:
        provider = first_connected_provider() or "claude"
        SettingsDialog(provider, PROVIDERS[provider]["default_model"], self).exec()

    def _continue(self) -> None:
        item = self.list.currentItem()
        if item is None:
            QMessageBox.information(self, "Drifter", "Pick a session, or create/import one.")
            return
        self.chosen_session_id = item.data(Qt.UserRole)
        self.accept()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    """Chat + live drift + corrective prompt + coach bar for one session."""

    def __init__(self, monitor: DriftMonitor, session_id: str) -> None:
        super().__init__()
        self.monitor = monitor
        self.session_id = session_id
        self.provider = first_connected_provider() or "claude"
        self.model = PROVIDERS[self.provider]["default_model"]
        self._thread: Optional[ChatThread] = None

        session = monitor.store.get_session(session_id)
        self.setWindowTitle(f"Drifter — {session.project_name if session else ''}")
        self.resize(1180, 760)

        self._build_ui(session)
        self._refresh_chart()
        self._update_coach()

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
        split.setSizes([580, 560])
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
            self._add_bubble(m.role, m.text)

        self.status = QLabel("")
        self.status.setObjectName("muted")
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setFixedHeight(76)
        self.input.setPlaceholderText("Message your AI…   (⌘/Ctrl + Enter to send)")
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        lay.addLayout(row)

        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self._on_send)
        QShortcut(QKeySequence("Meta+Return"), self.input, activated=self._on_send)
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
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        self.auto_check = QCheckBox("Auto re-align")
        self.auto_check.setChecked(True)
        self.clip_check = QCheckBox("Capture clipboard")
        self.clip_check.setChecked(is_watcher_running())
        self.clip_check.toggled.connect(self._on_clip_toggle)
        th_row.addWidget(th_lbl)
        th_row.addWidget(self.threshold_spin)
        th_row.addStretch(1)
        th_row.addWidget(self.auto_check)
        th_row.addWidget(self.clip_check)
        lay.addLayout(th_row)

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
    def _add_bubble(self, role: str, text: str) -> None:
        bubble = QLabel(text)
        bubble.setObjectName("bubbleUser" if (role or "").lower() == "user" else "bubbleAsst")
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(460)
        wrap = QHBoxLayout()
        if (role or "").lower() == "user":
            wrap.addStretch(1)
            wrap.addWidget(bubble)
        else:
            wrap.addWidget(bubble)
            wrap.addStretch(1)
        container = QWidget()
        container.setLayout(wrap)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, container)
        QTimer.singleShot(30, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()))

    # -- coach + provider label --------------------------------------------- #
    def _sync_provider_label(self) -> None:
        dot = "●" if get_key(self.provider) else "○"
        self.provider_label.setText(f"{dot} {PROVIDERS[self.provider]['label']} · {self.model}")

    def _update_coach(self) -> None:
        n_messages = len(self.monitor.store.get_messages(self.session_id))
        if not get_key(self.provider):
            self.coach.setText(
                f"① Connect your AI to start chatting — open Settings (top right), pick "
                f"{PROVIDERS[self.provider]['label']}, and paste an API key."
            )
        elif n_messages == 0:
            self.coach.setText("② You’re connected. Type your first message below and press Send.")
        elif self._last_drift_high():
            self.coach.setText("③ Drift detected — review the corrective prompt on the right and ‘Send to re-align’.")
        else:
            self.coach.setText("Monitoring — keep chatting. The chart updates live as drift changes.")

    def _last_drift_high(self) -> bool:
        ts = self.monitor.timeseries(self.session_id)
        anchor = ts.get("drift_from_anchor") or []
        return bool(anchor) and anchor[-1] > float(ts.get("threshold", self.monitor.threshold))

    # -- actions ------------------------------------------------------------- #
    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.provider, self.model, self)
        if dlg.exec() == QDialog.Accepted:
            provider, model, _ = dlg.result_values()
            self.provider = provider
            self.model = model or PROVIDERS[provider]["default_model"]
            self._sync_provider_label()
            self._update_coach()

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

    def _busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.status.setText("Thinking…" if busy else "")

    def _history(self) -> List[dict]:
        return [
            {"role": m.role, "content": m.text}
            for m in self.monitor.store.get_messages(self.session_id)
        ]

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or (self._thread and self._thread.isRunning()):
            return
        if not get_key(self.provider):
            QMessageBox.information(self, "Drifter", "Connect your AI in Settings first.")
            self._open_settings()
            return
        self.input.clear()
        res = self.monitor.add_turn(self.session_id, "user", text)
        self._add_bubble("user", text)
        self._refresh_chart()
        realign = bool(self.auto_check.isChecked() and res.get("alert"))
        self._busy(True)
        self._thread = ChatThread(
            self.provider, self.model, get_key(self.provider), self._history(), self._system_prompt(realign)
        )
        self._thread.done.connect(self._on_reply)
        self._thread.failed.connect(self._on_reply_error)
        self._thread.start()

    def _on_reply(self, reply: str) -> None:
        self._busy(False)
        if not reply:
            self.status.setText("(empty reply)")
            return
        self.monitor.add_turn(self.session_id, "assistant", reply)
        self._add_bubble("assistant", reply)
        self._refresh_chart()
        self._update_coach()

    def _on_reply_error(self, message: str) -> None:
        self._busy(False)
        QMessageBox.warning(self, "LLM error", message)

    def _copy_corrective(self) -> None:
        QApplication.clipboard().setText(self.corr_text.toPlainText())
        self.status.setText("Corrective prompt copied.")

    def _send_corrective(self) -> None:
        if (self._thread and self._thread.isRunning()) or not get_key(self.provider):
            if not get_key(self.provider):
                self._open_settings()
            return
        prompt = self.monitor.current_corrective_prompt(self.session_id)
        history = self._history() + [{"role": "user", "content": prompt}]
        self.monitor.add_turn(self.session_id, "user", prompt)
        self._add_bubble("user", "↻ Re-align: corrective prompt sent")
        self._refresh_chart()
        self._busy(True)
        self._thread = ChatThread(
            self.provider, self.model, get_key(self.provider), history, self._system_prompt(False)
        )
        self._thread.done.connect(self._on_reply)
        self._thread.failed.connect(self._on_reply_error)
        self._thread.start()

    # -- refresh ------------------------------------------------------------- #
    def _tick(self) -> None:
        self._refresh_chart()
        self._update_coach()

    def _refresh_chart(self) -> None:
        try:
            ts = self.monitor.timeseries(self.session_id)
        except Exception:
            return
        turns = ts.get("turns") or []
        anchor = ts.get("drift_from_anchor") or []
        reference = ts.get("drift_from_reference") or []
        threshold = float(ts.get("threshold", self.monitor.threshold))
        self.chart.update_series(turns, anchor, reference, threshold)

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
    app.setStyleSheet(QSS)
    app.setFont(QFont("-apple-system", 13))

    monitor = DriftMonitor()
    has_sessions = bool(monitor.store.list_sessions())

    # First run (no key, no sessions) -> guided onboarding; else session picker.
    if not any_key_present() and not has_sessions:
        dlg: QDialog = OnboardingWizard(monitor)
    else:
        dlg = LaunchDialog(monitor)
    if dlg.exec() != QDialog.Accepted or not dlg.chosen_session_id:
        return 0

    monitor.store.set_active_session(dlg.chosen_session_id)
    window = MainWindow(monitor, dlg.chosen_session_id)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

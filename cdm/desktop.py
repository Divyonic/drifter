"""Drifter — native desktop app (PySide6 + pyqtgraph).

A real desktop window (no browser): chat with your LLM in-app, watch context drift
climb live on a clean chart, and re-align with one click when it crosses the line.
Everything is local — sessions live in SQLite on this machine, API calls go straight
to the provider with a key stored locally, and the drift engine runs offline.

Launch with ``drifter`` (or ``python -m cdm.desktop``). Theme: white primary,
orange secondary, minimal/technical.
"""

from __future__ import annotations

import os
from typing import List, Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cdm import config
from cdm.llm import PROVIDERS, get_key, load_keys, save_key
from cdm.monitor import DriftMonitor
from cdm.transcript import parse_transcript
from cdm.watcher import is_watcher_running, start_watcher_process, stop_watcher

# --------------------------------------------------------------------------- #
# Palette + theme (white primary, orange secondary, minimal/technical)
# --------------------------------------------------------------------------- #
C = {
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "panel": "#FAFAFB",
    "ink": "#16181D",
    "muted": "#6B7280",
    "line": "#ECEDF0",
    "accent": "#FF6A1A",
    "accent_soft": "#FFF1E8",
    "danger": "#E11D48",
    "user_bubble": "#16181D",
    "assistant_bubble": "#F4F5F7",
}

QSS = f"""
* {{ font-family: -apple-system, "SF Pro Text", "Inter", "Segoe UI", sans-serif; }}
QWidget {{ background: {C['bg']}; color: {C['ink']}; font-size: 13px; }}
QLabel#h1 {{ font-size: 20px; font-weight: 700; }}
QLabel#h2 {{ font-size: 15px; font-weight: 600; }}
QLabel#muted {{ color: {C['muted']}; }}
QLabel#anchor {{ color: {C['muted']}; font-size: 12px; }}
QFrame#card {{ background: {C['panel']}; border: 1px solid {C['line']}; border-radius: 12px; }}
QFrame#divider {{ background: {C['line']}; max-height: 1px; min-height: 1px; border: none; }}

QPushButton {{
    background: {C['panel']}; color: {C['ink']}; border: 1px solid {C['line']};
    border-radius: 9px; padding: 8px 14px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {C['accent']}; }}
QPushButton#primary {{ background: {C['accent']}; color: #FFFFFF; border: none; }}
QPushButton#primary:hover {{ background: #FF7E38; }}
QPushButton#ghost {{ background: transparent; border: none; color: {C['muted']}; padding: 4px 8px; }}
QPushButton#ghost:hover {{ color: {C['accent']}; }}

QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox {{
    background: #FFFFFF; border: 1px solid {C['line']}; border-radius: 9px; padding: 8px 10px;
    selection-background-color: {C['accent_soft']};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {C['accent']}; }}
QComboBox::drop-down {{ border: none; }}

QListWidget {{ background: #FFFFFF; border: 1px solid {C['line']}; border-radius: 12px; padding: 6px; }}
QListWidget::item {{ padding: 12px 12px; border-radius: 9px; }}
QListWidget::item:selected {{ background: {C['accent_soft']}; color: {C['ink']}; }}
QListWidget::item:hover {{ background: {C['panel']}; }}

QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #D7DAE0; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QLabel#chipOk {{ background: {C['accent_soft']}; color: {C['accent']}; border-radius: 9px; padding: 5px 12px; font-weight: 700; }}
QLabel#chipBad {{ background: #FDE7EC; color: {C['danger']}; border-radius: 9px; padding: 5px 12px; font-weight: 700; }}
QLabel#bubbleUser {{ background: {C['user_bubble']}; color: #FFFFFF; border-radius: 14px; padding: 10px 13px; }}
QLabel#bubbleAsst {{ background: {C['assistant_bubble']}; color: {C['ink']}; border-radius: 14px; padding: 10px 13px; }}
QCheckBox {{ color: {C['muted']}; }}
"""


def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    return f


# --------------------------------------------------------------------------- #
# Local profile (offline; just a display name — nothing leaves the PC)
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


# --------------------------------------------------------------------------- #
# Drift chart
# --------------------------------------------------------------------------- #
class DriftChart(pg.PlotWidget):
    """Minimal real-time drift plot: orange = vs anchor, grey dashed = vs reference."""

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
            self.getAxis(ax).setPen(C["line"])
        self.getAxis("left").setLabel("drift", color=C["muted"])
        self.getAxis("bottom").setLabel("turn", color=C["muted"])

        self._anchor = self.plot(
            [], [], pen=pg.mkPen(C["accent"], width=3),
            symbol="o", symbolSize=5, symbolBrush=C["accent"], symbolPen=None,
        )
        self._reference = self.plot(
            [], [], pen=pg.mkPen("#C7CBD1", width=2, style=Qt.DashLine)
        )
        self._threshold = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(C["danger"], width=1, style=Qt.DotLine)
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
    """Run one LLM request off the UI thread."""

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, provider, model, api_key, messages, system) -> None:
        super().__init__()
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._messages = messages
        self._system = system

    def run(self) -> None:  # noqa: D401
        try:
            from cdm.llm import LLMClient

            client = LLMClient(self._provider, api_key=self._api_key, model=self._model)
            self.done.emit(client.chat(self._messages, system=self._system))
        except Exception as exc:
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------- #
# Dialogs
# --------------------------------------------------------------------------- #
class NewSessionDialog(QDialog):
    """Collect project name, anchor goal and constraints for a new session."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New session")
        self.setMinimumWidth(440)
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.goal = QPlainTextEdit()
        self.goal.setFixedHeight(80)
        self.constraints = QPlainTextEdit()
        self.constraints.setFixedHeight(70)
        self.constraints.setPlaceholderText("One per line, e.g. under 5 kg")
        form.addRow("Project", self.name)
        form.addRow("Goal (anchor)", self.goal)
        form.addRow("Constraints", self.constraints)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        cons = [c.strip() for c in self.constraints.toPlainText().splitlines() if c.strip()]
        return self.name.text().strip(), self.goal.toPlainText().strip(), cons


class SettingsDialog(QDialog):
    """Pick the active provider, model and (locally stored) API key."""

    def __init__(self, provider: str, model: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LLM settings")
        self.setMinimumWidth(440)
        form = QFormLayout(self)

        self.provider = QComboBox()
        for key, meta in PROVIDERS.items():
            self.provider.addItem(meta["label"], key)
        idx = list(PROVIDERS).index(provider) if provider in PROVIDERS else 0
        self.provider.setCurrentIndex(idx)

        self.model = QLineEdit(model)
        self.key = QLineEdit(load_keys().get(provider, ""))
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Stored locally only")

        hint = QLabel("Keys are saved on this machine (no server). Calls go directly to the provider.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)

        form.addRow("Provider", self.provider)
        form.addRow("Model", self.model)
        form.addRow("API key", self.key)
        form.addRow(hint)

        self.provider.currentIndexChanged.connect(self._on_provider)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_provider(self) -> None:
        key = self.provider.currentData()
        self.model.setText(PROVIDERS[key]["default_model"])
        self.key.setText(load_keys().get(key, ""))

    def result_values(self):
        return self.provider.currentData(), self.model.text().strip(), self.key.text().strip()


class LaunchDialog(QDialog):
    """Startup screen: greet the local profile, list past sessions, pick one."""

    def __init__(self, monitor: DriftMonitor, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.chosen_session_id: Optional[str] = None
        self.setWindowTitle("Drifter")
        self.setMinimumSize(560, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(14)

        name = load_profile_name()
        title = QLabel("Drifter")
        title.setObjectName("h1")
        root.addWidget(title)

        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Your name (stored locally)")
        if not name:
            row = QHBoxLayout()
            row.addWidget(QLabel("Welcome —"))
            row.addWidget(self.name_edit)
            root.addLayout(row)

        prompt = QLabel("Which session do you want to continue today?")
        prompt.setObjectName("h2")
        root.addWidget(prompt)

        self.list = QListWidget()
        self._populate()
        self.list.itemDoubleClicked.connect(lambda _i: self._continue())
        root.addWidget(self.list, 1)

        btns = QHBoxLayout()
        new_btn = QPushButton("New session")
        imp_btn = QPushButton("Import chat…")
        cont_btn = QPushButton("Continue")
        cont_btn.setObjectName("primary")
        new_btn.clicked.connect(self._new)
        imp_btn.clicked.connect(self._import)
        cont_btn.clicked.connect(self._continue)
        btns.addWidget(new_btn)
        btns.addWidget(imp_btn)
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
            item = QListWidgetItem(f"{s.project_name}\n{n} turns · {s.updated_at[:16].replace('T', ' ')}")
            item.setData(Qt.UserRole, s.session_id)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _save_name(self) -> None:
        nm = self.name_edit.text().strip()
        if nm and nm != load_profile_name():
            save_profile_name(nm)

    def _new(self) -> None:
        dlg = NewSessionDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name, goal, cons = dlg.values()
            if not goal:
                QMessageBox.warning(self, "Drifter", "A goal is required.")
                return
            session = self.monitor.start_session(name or "Untitled", goal, cons)
            self._save_name()
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
        anchor = turns[0]["text"]
        session = self.monitor.start_session("Imported chat", anchor, [])
        if len(turns) > 1:
            self.monitor.ingest_transcript(session.session_id, turns[1:])
        self._save_name()
        self.chosen_session_id = session.session_id
        self.accept()

    def _continue(self) -> None:
        item = self.list.currentItem()
        if item is None:
            QMessageBox.information(self, "Drifter", "Pick a session, or create/import one.")
            return
        self._save_name()
        self.chosen_session_id = item.data(Qt.UserRole)
        self.accept()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    """Chat + live drift + corrective prompt for a single session."""

    def __init__(self, monitor: DriftMonitor, session_id: str) -> None:
        super().__init__()
        self.monitor = monitor
        self.session_id = session_id
        self.provider = "claude"
        self.model = PROVIDERS[self.provider]["default_model"]
        self._thread: Optional[ChatThread] = None

        session = monitor.store.get_session(session_id)
        self.setWindowTitle(f"Drifter — {session.project_name if session else ''}")
        self.resize(1120, 720)

        self._build_ui(session)
        self._refresh_chart()

        # Live refresh (covers clipboard-sourced turns + keeps chart in sync).
        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh_chart)
        self._timer.start()

    # -- layout -------------------------------------------------------------- #
    def _build_ui(self, session) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(12)

        # Header row
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
        outer.addWidget(_divider())

        # Split: chat (left) | drift (right)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_chat_panel())
        split.addWidget(self._build_drift_panel())
        split.setSizes([560, 540])
        outer.addWidget(split, 1)

        self._sync_provider_label()

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
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

        # render existing history
        for m in self.monitor.store.get_messages(self.session_id):
            self._add_bubble(m.role, m.text)

        self.status = QLabel("")
        self.status.setObjectName("muted")
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setFixedHeight(72)
        self.input.setPlaceholderText("Message your LLM…  (Ctrl+Enter to send)")
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        lay.addLayout(row)

        # Ctrl+Enter to send
        from PySide6.QtGui import QShortcut, QKeySequence

        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self._on_send)
        QShortcut(QKeySequence("Meta+Return"), self.input, activated=self._on_send)
        return panel

    def _build_drift_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # status + metrics row
        top = QHBoxLayout()
        self.chip = QLabel("on track")
        self.chip.setObjectName("chipOk")
        self.metric = QLabel("drift 0.000")
        self.metric.setObjectName("muted")
        top.addWidget(self.chip)
        top.addStretch(1)
        top.addWidget(self.metric)
        lay.addLayout(top)

        self.chart = DriftChart()
        self.chart.setMinimumHeight(260)
        lay.addWidget(self.chart, 1)

        # threshold control
        th_row = QHBoxLayout()
        th_lbl = QLabel("Threshold")
        th_lbl.setObjectName("muted")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.30, 0.95)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(round(self.monitor.threshold, 2))
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        self.clip_check = QCheckBox("Also capture clipboard")
        self.clip_check.setChecked(is_watcher_running())
        self.clip_check.toggled.connect(self._on_clip_toggle)
        self.auto_check = QCheckBox("Auto re-align when drifting")
        self.auto_check.setChecked(True)
        th_row.addWidget(th_lbl)
        th_row.addWidget(self.threshold_spin)
        th_row.addStretch(1)
        th_row.addWidget(self.auto_check)
        th_row.addWidget(self.clip_check)
        lay.addLayout(th_row)

        # corrective card
        self.corr_card = QFrame()
        self.corr_card.setObjectName("card")
        cc = QVBoxLayout(self.corr_card)
        cc.setContentsMargins(14, 12, 14, 12)
        cc_title = QLabel("Corrective prompt")
        cc_title.setObjectName("h2")
        cc.addWidget(cc_title)
        self.corr_text = QPlainTextEdit()
        self.corr_text.setReadOnly(True)
        self.corr_text.setFixedHeight(120)
        cc.addWidget(self.corr_text)
        cc_btns = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy_corrective)
        send_corr = QPushButton("Send to re-align")
        send_corr.setObjectName("primary")
        send_corr.clicked.connect(self._send_corrective)
        cc_btns.addStretch(1)
        cc_btns.addWidget(copy_btn)
        cc_btns.addWidget(send_corr)
        cc.addLayout(cc_btns)
        lay.addWidget(self.corr_card)
        self.corr_card.setVisible(False)

        return panel

    # -- chat bubbles -------------------------------------------------------- #
    def _add_bubble(self, role: str, text: str) -> None:
        bubble = QLabel(text)
        bubble.setObjectName("bubbleUser" if (role or "").lower() == "user" else "bubbleAsst")
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(440)
        wrap = QHBoxLayout()
        if (role or "").lower() == "user":
            wrap.addStretch(1)
            wrap.addWidget(bubble)
        else:
            wrap.addWidget(bubble)
            wrap.addStretch(1)
        container = QWidget()
        container.setLayout(wrap)
        # insert before the trailing stretch
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, container)
        QTimer.singleShot(30, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()))

    # -- actions ------------------------------------------------------------- #
    def _sync_provider_label(self) -> None:
        has_key = bool(get_key(self.provider))
        dot = "●" if has_key else "○"
        self.provider_label.setText(f"{dot} {PROVIDERS[self.provider]['label']} · {self.model}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.provider, self.model, self)
        if dlg.exec() == QDialog.Accepted:
            provider, model, key = dlg.result_values()
            self.provider = provider
            self.model = model or PROVIDERS[provider]["default_model"]
            if key:
                save_key(provider, key)
            self._sync_provider_label()

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

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or (self._thread and self._thread.isRunning()):
            return
        if not get_key(self.provider):
            QMessageBox.information(self, "Drifter", "Add an API key in Settings first.")
            self._open_settings()
            return

        self.input.clear()
        res = self.monitor.add_turn(self.session_id, "user", text)
        self._add_bubble("user", text)
        self._refresh_chart()
        realign = bool(self.auto_check.isChecked() and res.get("alert"))

        history = [
            {"role": m.role, "content": m.text}
            for m in self.monitor.store.get_messages(self.session_id)
        ]
        self._busy(True)
        self._thread = ChatThread(
            self.provider, self.model, get_key(self.provider), history, self._system_prompt(realign)
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

    def _on_reply_error(self, message: str) -> None:
        self._busy(False)
        QMessageBox.warning(self, "LLM error", message)

    def _copy_corrective(self) -> None:
        QApplication.clipboard().setText(self.corr_text.toPlainText())
        self.status.setText("Corrective prompt copied.")

    def _send_corrective(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        if not get_key(self.provider):
            self._open_settings()
            return
        prompt = self.monitor.current_corrective_prompt(self.session_id)
        history = [
            {"role": m.role, "content": m.text}
            for m in self.monitor.store.get_messages(self.session_id)
        ]
        history.append({"role": "user", "content": prompt})
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

    # -- chart refresh ------------------------------------------------------- #
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
        # re-apply stylesheet for objectName change to take effect
        self.chip.setStyleSheet(self.chip.styleSheet())
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

    launch = LaunchDialog(monitor)
    if launch.exec() != QDialog.Accepted or not launch.chosen_session_id:
        return 0
    monitor.store.set_active_session(launch.chosen_session_id)

    window = MainWindow(monitor, launch.chosen_session_id)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

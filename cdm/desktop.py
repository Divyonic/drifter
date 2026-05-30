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
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
from PySide6.QtCore import (
    QEasingCurve,
    QEventLoop,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Property,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
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
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QSplashScreen,
    QStackedWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cdm import claude_code as cc
from cdm import config
from cdm.corrective import strictness_line
from cdm.embeddings import fastembed_available, get_embedder
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

# Live MainWindows, so an automatic OS light/dark switch can re-pen their charts and
# legend swatches (QSS alone doesn't repaint pyqtgraph pens or painted pixmaps).
_LIVE_WINDOWS: list = []


def set_dark(dark: bool) -> None:
    """Switch the active palette."""
    C.clear()
    C.update(DARK if dark else LIGHT)


_QSS = """
* { font-family: "SF Mono", "JetBrains Mono", "Menlo", "Cascadia Code", "Consolas", monospace; }
QWidget { background: @bg@; color: @ink@; font-size: 13px; }
QMainWindow, QDialog { background: @bg@; }
QLabel#h1 { font-size: 26px; font-weight: 600; }
QLabel#h2 { font-size: 17px; font-weight: 600; }
QLabel#title { font-size: 18px; font-weight: 600; color: @ink@; }
QLabel#muted { color: @muted@; }
QLabel#anchor { color: @muted@; font-size: 12px; }
QLabel#providerChip { color: @muted@; font-size: 12px; font-weight: 600; background: @panel@; border: 1px solid @line_soft@; border-radius: 11px; padding: 5px 11px; }
QFrame#vsep { background: @line_soft@; border: none; }
QPushButton#iconBtn { background: transparent; border: none; border-radius: 9px; font-size: 17px; padding: 0; }
QPushButton#iconBtn:hover { background: @hover@; }
QLabel#coach { background: @coach_bg@; color: @coach_fg@; border-radius: 12px; padding: 11px 15px; font-weight: 600; }
QLabel#warn { color: @coach_fg@; font-size: 12px; font-weight: 600; padding: 2px 2px; }
QLabel#sectionLabel { color: @muted@; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QFrame#card { background: @bg@; border: 1px solid @line_soft@; border-radius: 16px; }
QFrame#toolbar { background: @panel@; border: 1px solid @line_soft@; border-radius: 14px; }
QFrame#legendStrip { background: transparent; border: none; }
QLabel#legendSwatch { background: transparent; border: none; padding: 0; }
QLabel#legendKey { color: @muted@; font-size: 12px; background: transparent; border: none; }
QFrame#hairline { background: @line@; max-height: 1px; min-height: 1px; border: none; }
/* Single-line controls: NO vertical padding (it makes macOS clip the text); height
   comes from min-height so the text always has full room and centres vertically. */
QPushButton { background: @bg@; color: @ink@; border: 1px solid @line@; border-radius: 10px; padding: 0 16px; min-height: 34px; font-weight: 600; }
QPushButton:hover { background: @hover@; }
QPushButton:disabled { color: @muted@; border-color: @line_soft@; }
QPushButton#primary { background: @accent@; color: #FFFFFF; border: none; }
QPushButton#primary:hover { background: @accent_hover@; }
QPushButton#primary:disabled { background: @line@; }
QPushButton#link { background: transparent; border: none; color: @accent@; padding: 0 6px; min-height: 30px; font-weight: 600; }
QPushButton#link:hover { color: @accent_hover@; }
QPushButton#ghost { background: transparent; border: 1px solid @line_soft@; border-radius: 9px; padding: 0 12px; min-height: 32px; font-weight: 600; }
QPushButton#ghost:hover { background: @hover@; }
QPushButton#seg { background: @panel@; color: @muted@; border: 1px solid @line_soft@; border-radius: 9px; padding: 0 16px; min-height: 32px; font-weight: 600; }
QPushButton#seg:hover { color: @ink@; }
QPushButton#segOn { background: @bg@; color: @ink@; border: 1px solid @accent@; border-radius: 9px; padding: 0 16px; min-height: 32px; font-weight: 700; }
QLineEdit, QComboBox, QDoubleSpinBox { background: @input@; color: @ink@; border: 1px solid @line@; border-radius: 10px; padding: 0 11px; min-height: 34px; selection-background-color: @sel@; selection-color: @ink@; }
QPlainTextEdit { background: @input@; color: @ink@; border: 1px solid @line@; border-radius: 10px; padding: 8px 11px; selection-background-color: @sel@; selection-color: @ink@; }
QComboBox QLineEdit { background: transparent; color: @ink@; border: none; padding: 0; min-height: 0; }
QComboBox::item { min-height: 28px; }
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
QFrame#statCard { background: @panel@; border: 1px solid @line_soft@; border-radius: 14px; }
QLabel#iconChip { background: @bg@; border: 1px solid @line_soft@; border-radius: 8px; color: @ink@; font-size: 13px; }
QLabel#statTitle { color: @muted@; font-size: 12px; font-weight: 600; }
QLabel#statValue { color: @ink@; font-size: 24px; font-weight: 700; }
QLabel#statCaption { color: @muted@; font-size: 11px; }
QLabel#statTrend { color: @muted@; font-size: 11px; font-weight: 700; }
QLabel#statTrend[good="1"] { color: @okfg@; }
QLabel#statTrend[good="0"] { color: @danger@; }
QLabel#pillOk { background: @okbg@; color: @okfg@; border-radius: 9px; padding: 4px 11px; font-size: 11px; font-weight: 700; }
QLabel#pillBad { background: @badbg@; color: @badfg@; border-radius: 9px; padding: 4px 11px; font-size: 11px; font-weight: 700; }
QLabel#pillWarn { background: @coach_bg@; color: @coach_fg@; border-radius: 9px; padding: 4px 11px; font-size: 11px; font-weight: 700; }
QLabel#bubbleUser { background: @ubg@; color: @ufg@; border-radius: 18px; padding: 11px 14px; }
QLabel#bubbleAsst { background: @abg@; color: @afg@; border-radius: 18px; padding: 11px 14px; }
QCheckBox { color: @ink@; spacing: 8px; background: transparent; }
QCheckBox:disabled { color: @muted@; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid @line@; border-radius: 5px; background: @input@; }
QCheckBox::indicator:hover { border: 1px solid @accent@; }
QCheckBox::indicator:checked { background: @accent@; border: 1px solid @accent@; image: url(@checkurl@); }
QCheckBox::indicator:checked:hover { background: @accent_hover@; border: 1px solid @accent_hover@; }
"""


def build_qss() -> str:
    """Render the stylesheet against the active palette ``C``."""
    s = _QSS
    for key, value in C.items():
        s = s.replace(f"@{key}@", value)
    # A bundled white tick for the checked checkbox. Qt's QSS url() wants a plain
    # absolute path (a file:// URI gets treated as relative); forward slashes are safe.
    s = s.replace("@checkurl@", _asset("check.svg").replace("\\", "/"))
    return s


def is_dark() -> bool:
    """True if the dark palette is currently active."""
    return C.get("bg") == DARK["bg"]


def _apply_theme(dark: bool, store=None) -> None:
    """Pin light/dark and restyle live (kept for callers that think in booleans)."""
    apply_theme_choice("dark" if dark else "light", store)


def system_is_dark() -> bool:
    """Best-effort read of the OS colour scheme (defaults to light)."""
    try:
        return QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        return False


def apply_theme_choice(choice: str, store=None) -> None:
    """Apply an explicit appearance choice: ``auto`` / ``light`` / ``dark``.

    ``auto`` follows the OS now and keeps following it (persisted as ``auto`` so the
    live colour-scheme listener stays active); ``light`` / ``dark`` pin the palette.
    """
    choice = choice if choice in ("auto", "light", "dark") else "auto"
    dark = system_is_dark() if choice == "auto" else (choice == "dark")
    set_dark(dark)
    app = QApplication.instance()
    if app is not None:
        tops = app.topLevelWidgets()
        for w in tops:
            w.setUpdatesEnabled(False)
        try:
            app.setStyleSheet(build_qss())
        finally:
            for w in tops:
                w.setUpdatesEnabled(True)
    if store is not None:
        try:
            store.set_meta("theme", choice)
        except Exception:
            pass


def _md_to_html(text: str) -> str:
    """Render markdown to the HTML subset QLabel understands (with a plain fallback)."""
    try:
        from markdown_it import MarkdownIt

        return MarkdownIt("commonmark", {"breaks": True}).render(text or "")
    except Exception:
        import html as _html

        return "<span>" + _html.escape(text or "").replace("\n", "<br>") + "</span>"


def _rgba(hex_color: str, alpha: int) -> QColor:
    """A QColor from a hex string with an explicit alpha (0-255)."""
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


def _legend_swatch(kind: str, dpr: float = 2.0) -> QPixmap:
    """A small pixmap that reproduces a chart mark exactly, for the legend strip.

    Reads the live ``C`` palette so a theme switch just re-calls it. The pens mirror
    :class:`DriftChart` one-for-one so the key can never lie about the chart:
    ``goal`` solid accent line + dot · ``recent`` dashed muted line · ``threshold``
    red dotted line · ``band`` translucent rounded rect.
    """
    w, h = 28, 14
    pm = QPixmap(int(w * dpr), int(h * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)  # the card background shows through (works light + dark)
    p = QPainter(pm)
    cy = h / 2.0
    try:
        if kind == "goal":
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(QColor(C["accent"]), 2.2))
            p.drawLine(QPointF(2, cy), QPointF(w - 2, cy))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C["accent"]))
            p.drawEllipse(QPointF(w / 2.0, cy), 2.5, 2.5)
        elif kind == "recent":
            p.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(_rgba(C["muted"], 150), 1.0)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(2, cy), QPointF(w - 2, cy))
        elif kind == "threshold":
            p.setRenderHint(QPainter.Antialiasing, False)  # keep dots crisp
            pen = QPen(QColor(C["danger"]), 1.4)
            pen.setStyle(Qt.DotLine)
            p.setPen(pen)
            p.drawLine(QPointF(2, cy), QPointF(w - 2, cy))
        elif kind == "band":
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(_rgba(C["line"], 90), 1))
            p.setBrush(_rgba(C["muted"], 30))
            p.drawRoundedRect(QRectF(2, 3, w - 4, h - 6), 3, 3)
    finally:
        p.end()
    return pm


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


def _vsep() -> QFrame:
    """A short vertical hairline divider for toolbars/headers."""
    f = QFrame()
    f.setObjectName("vsep")
    f.setFixedWidth(1)
    f.setFixedHeight(20)
    return f


class ElidingLabel(QLabel):
    """A QLabel that truncates with an ellipsis to fit its width (full text in tooltip).

    Lets a long session title sit in the header without overflowing or clipping —
    it elides on the right and re-elides as the window resizes.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._full = text
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def setText(self, text: str) -> None:  # noqa: D401
        self._full = text or ""
        self.setToolTip(self._full)
        super().setText(self._full)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        fm = self.fontMetrics()
        elided = fm.elidedText(self._full, Qt.ElideRight, self.width())
        painter.setPen(QColor(C["ink"]))  # follows the active light/dark palette
        painter.drawText(self.rect(), int(self.alignment() | Qt.AlignVCenter), elided)


def _asset(name: str) -> str:
    """Absolute path to a bundled asset under cdm/assets/."""
    return str(Path(__file__).resolve().parent / "assets" / name)


def logo_pixmap(height: int = 28) -> QPixmap:
    """The [DRIFTER] logo scaled to ``height`` px (empty pixmap if missing)."""
    pm = QPixmap(_asset("drifter_logo.png"))
    if pm.isNull():
        return pm
    return pm.scaledToHeight(int(height), Qt.SmoothTransformation)


def logo_label(height: int = 26) -> QWidget:
    """A fixed top-left brand mark: the logo, or a text fallback."""
    lab = QLabel()
    pm = logo_pixmap(height)
    if not pm.isNull():
        lab.setPixmap(pm)
    else:
        lab.setText("[DRIFTER]")
        lab.setObjectName("h2")
    return lab


def show_splash(app: QApplication) -> None:
    """Show the logo splash, fade it in then out, and return when done."""
    pm = logo_pixmap(height=150)
    if pm.isNull():
        return
    canvas = QPixmap(560, 340)
    canvas.fill(QColor("#FFFFFF"))  # the logo is designed for a white field
    painter = QPainter(canvas)
    painter.drawPixmap((560 - pm.width()) // 2, (340 - pm.height()) // 2, pm)
    painter.end()

    splash = QSplashScreen(canvas)
    splash.setWindowOpacity(0.0)
    splash.show()
    app.processEvents()

    fade_in = QPropertyAnimation(splash, b"windowOpacity")
    fade_in.setDuration(350)
    fade_in.setStartValue(0.0)
    fade_in.setEndValue(1.0)
    fade_in.start()
    splash._fade_in = fade_in  # keep a reference so it isn't GC'd

    loop = QEventLoop()

    def _fade_out() -> None:
        fade_out = QPropertyAnimation(splash, b"windowOpacity")
        fade_out.setDuration(550)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.finished.connect(loop.quit)
        fade_out.start()
        splash._fade_out = fade_out

    QTimer.singleShot(1000, _fade_out)
    loop.exec()
    splash.close()


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

    # Show per-point markers only for short conversations (clutter otherwise).
    _MARKER_LIMIT = 30

    def __init__(self) -> None:
        super().__init__()
        self.setBackground(C["bg"])
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)   # scroll to zoom, drag to pan
        self.setLimits(yMin=-0.05, yMax=2.1)
        self.hideButtons()
        self.setYRange(0, 2.0, padding=0.02)   # 0–2 cosine range; 1.0 sits centered
        self.showGrid(x=False, y=True, alpha=0.08)
        for ax in ("left", "bottom"):
            self.getAxis(ax).setTextPen(C["muted"])
            self.getAxis(ax).setPen(C["line_soft"])
        self.getAxis("left").setLabel("drift", color=C["muted"])
        self.getAxis("bottom").setLabel("turn", color=C["muted"])

        dgr = QColor(C["danger"])
        self._danger = pg.LinearRegionItem(
            orientation="horizontal", movable=False,
            brush=pg.mkBrush(dgr.red(), dgr.green(), dgr.blue(), 18), pen=pg.mkPen(None),
        )
        self._danger.setZValue(-20)
        self.addItem(self._danger)

        mut = QColor(C["muted"])
        self._band = pg.LinearRegionItem(
            orientation="horizontal", movable=False,
            brush=pg.mkBrush(mut.red(), mut.green(), mut.blue(), 22), pen=pg.mkPen(None),
        )
        self._band.setZValue(-10)
        self.addItem(self._band)
        self._band.hide()

        self._anchor = self.plot([], [], pen=pg.mkPen(C["accent"], width=2.2))
        self._anchor.setClipToView(True)
        self._anchor.setDownsampling(auto=True)
        self._reference = self.plot(
            [], [], pen=pg.mkPen(_rgba(C["muted"], 150), width=1.0, style=Qt.DashLine))
        self._reference.setClipToView(True)
        self._reference.setDownsampling(auto=True)
        self._forecast = self.plot([], [], pen=pg.mkPen(C["accent"], width=1.4, style=Qt.DashLine))
        self._threshold = pg.InfiniteLine(angle=0, pen=pg.mkPen(C["danger"], width=1.2, style=Qt.DotLine))
        self.addItem(self._threshold)
        self._cp = pg.InfiniteLine(angle=90, pen=pg.mkPen(C["muted"], width=1.0, style=Qt.DashLine))
        self.addItem(self._cp)
        self._cp.hide()
        self._fc_text = pg.TextItem(color=C["accent"], anchor=(0, 1))
        self.addItem(self._fc_text)
        self._fc_text.hide()

        # Hover crosshair + info tooltip.
        self._cross_v = pg.InfiniteLine(angle=90, pen=pg.mkPen(_rgba(C["muted"], 110), width=1))
        self._cross_h = pg.InfiniteLine(angle=0, pen=pg.mkPen(_rgba(C["muted"], 110), width=1))
        for ln in (self._cross_v, self._cross_h):
            ln.setZValue(40)
            self.addItem(ln)
            ln.hide()
        self._hover = pg.TextItem(
            anchor=(0, 1), color=C["ink"],
            fill=_rgba(C["panel"], 240), border=pg.mkPen(_rgba(C["line"], 255)),
        )
        self._hover.setZValue(60)
        self.addItem(self._hover)
        self._hover.hide()
        # series cache for hover lookups
        self._turns: list = []
        self._avals: list = []
        self._rvals: list = []
        self._roles: list = []
        self._texts: list = []
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _on_mouse_moved(self, pos) -> None:
        if not self._turns or not self.sceneBoundingRect().contains(pos):
            self._cross_v.hide()
            self._cross_h.hide()
            self._hover.hide()
            return
        mp = self.getViewBox().mapSceneToView(pos)
        x = mp.x()
        idx = min(range(len(self._turns)), key=lambda i: abs(self._turns[i] - x))
        turn = self._turns[idx]
        a = self._avals[idx] if idx < len(self._avals) else 0.0
        r = self._rvals[idx] if idx < len(self._rvals) else 0.0
        role = self._roles[idx] if idx < len(self._roles) else ""
        text = self._texts[idx] if idx < len(self._texts) else ""
        snippet = (text[:52] + "…") if len(text) > 52 else text
        self._cross_v.setPos(turn)
        self._cross_h.setPos(a)
        self._cross_v.show()
        self._cross_h.show()
        # flip the tooltip to the left near the right edge so it stays on-screen
        rng = self.getViewBox().viewRange()[0]
        anchor_x = 1.0 if (rng[1] - turn) < (rng[1] - rng[0]) * 0.35 else 0.0
        self._hover.setAnchor((anchor_x, 1))
        self._hover.setText(
            f"turn {turn} · {role}\ndrift {a:.2f}  ·  ref {r:.2f}"
            + (f"\n{snippet}" if snippet else "")
        )
        self._hover.setPos(turn, min(2.0, a + 0.04))
        self._hover.show()

    def apply_theme(self) -> None:
        """Recolor the chart for the current palette (used on theme toggle)."""
        try:
            self.setBackground(C["bg"])
            for ax in ("left", "bottom"):
                self.getAxis(ax).setTextPen(C["muted"])
                self.getAxis(ax).setPen(C["line_soft"])
            self._anchor.setPen(pg.mkPen(C["accent"], width=2.2))
            self._reference.setPen(pg.mkPen(_rgba(C["muted"], 150), width=1.0, style=Qt.DashLine))
            self._forecast.setPen(pg.mkPen(C["accent"], width=1.4, style=Qt.DashLine))
            self._threshold.setPen(pg.mkPen(C["danger"], width=1.2, style=Qt.DotLine))
            self._cp.setPen(pg.mkPen(C["muted"], width=1.0, style=Qt.DashLine))
            self._cross_v.setPen(pg.mkPen(_rgba(C["muted"], 110), width=1))
            self._cross_h.setPen(pg.mkPen(_rgba(C["muted"], 110), width=1))
            self._danger.setBrush(_rgba(C["danger"], 18))
            self._band.setBrush(_rgba(C["muted"], 22))
        except Exception:
            pass

    def reset_view(self) -> None:
        """Reset zoom/pan to the full series with the standard 0–1 drift range."""
        try:
            self.getViewBox().autoRange(padding=0.04)
        except Exception:
            pass
        self.setYRange(0, 2.0, padding=0.02)

    def update(self, ts: dict) -> None:
        turns = list(ts.get("turns") or [])
        anchor = list(ts.get("drift_from_anchor") or [])
        reference = list(ts.get("drift_from_reference") or [])
        threshold = float(ts.get("threshold", 0.65))
        self._turns, self._avals, self._rvals = turns, anchor, reference
        self._roles = list(ts.get("roles") or [])
        self._texts = list(ts.get("texts") or [])

        # markers only when few points, so a long chat stays clean
        if turns and len(turns) <= self._MARKER_LIMIT:
            self._anchor.setData(turns, anchor, symbol="o", symbolSize=5,
                                 symbolBrush=C["accent"], symbolPen=None)
        else:
            self._anchor.setData(turns, anchor, symbol=None)
        self._reference.setData(turns, reference)
        self._threshold.setValue(threshold)
        self._danger.setRegion([threshold, 2.1])  # subtle "off-track" zone (to top)

        # Baseline band only when it's tight enough to be meaningful (else it floods).
        mean, std = ts.get("baseline_mean"), ts.get("baseline_std")
        if mean is not None and std is not None and std <= 0.18 and len(turns) >= 3:
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
            self._fc_text.setPos(x0, min(2.0, threshold + 0.07))
            self._fc_text.show()
        else:
            self._forecast.setData([], [])
            self._fc_text.hide()


# --------------------------------------------------------------------------- #
# Dashboard primitives (painter widgets): gauge, sparkline, stat card, pill
# --------------------------------------------------------------------------- #
def _drift_color(value: float, threshold: float) -> str:
    """Status colour for a drift value: green on-goal, amber nearing, red over."""
    if value >= threshold:
        return C["danger"]
    if threshold > 0 and value >= threshold * 0.8:
        return C["accent"]
    return C["okfg"]


class DriftGauge(QWidget):
    """An animated radial drift gauge (reference: the 'Calories' arc gauge).

    Sweeps a ~250° arc over the 0–2 drift scale: a faint track, fine tick marks, a
    coloured fill up to the (animated) current drift, a notch at the threshold, and a
    big centre value + caption. Colour follows :func:`_drift_color`.
    """

    _START = 235.0      # degrees (Qt: 0=3 o'clock, CCW+); arc opens at the bottom
    _SWEEP = -290.0     # clockwise
    _MAXV = 2.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(150, 130)
        self._value = 0.0
        self._threshold = 0.65
        self._caption = "on track"
        self._anim = QPropertyAnimation(self, b"value", self)
        self._anim.setDuration(520)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _get_value(self) -> float:
        return self._value

    def _set_value(self, v: float) -> None:
        self._value = float(v)
        self.update()

    value = Property(float, _get_value, _set_value)

    def set_state(self, drift: float, threshold: float, caption: str = "") -> None:
        self._threshold = float(threshold)
        self._caption = caption or self._caption
        target = max(0.0, min(self._MAXV, float(drift)))
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(target)
        self._anim.start()

    def apply_theme(self) -> None:
        self.update()

    def _frac(self, v: float) -> float:
        return max(0.0, min(1.0, v / self._MAXV))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        side = min(w, h * 1.18)
        pad = 16
        rect = QRectF((w - side) / 2 + pad, (h - side) / 2 + pad + 6,
                      side - 2 * pad, side - 2 * pad)
        # Track.
        track = QPen(_rgba(C["line"], 150), 9)
        track.setCapStyle(Qt.RoundCap)
        p.setPen(track)
        p.drawArc(rect, int(self._START * 16), int(self._SWEEP * 16))
        # Tick marks around the arc.
        import math
        cx, cy = rect.center().x(), rect.center().y()
        r_out = rect.width() / 2 + 3
        r_in = r_out - 6
        p.setPen(QPen(_rgba(C["muted"], 120), 1))
        for i in range(31):
            a = math.radians(self._START + self._SWEEP * (i / 30.0))
            p.drawLine(QPointF(cx + r_in * math.cos(a), cy - r_in * math.sin(a)),
                       QPointF(cx + r_out * math.cos(a), cy - r_out * math.sin(a)))
        # Coloured fill up to value.
        col = QColor(_drift_color(self._value, self._threshold))
        fill = QPen(col, 9)
        fill.setCapStyle(Qt.RoundCap)
        p.setPen(fill)
        p.drawArc(rect, int(self._START * 16), int(self._SWEEP * self._frac(self._value) * 16))
        # Threshold notch.
        ta = math.radians(self._START + self._SWEEP * self._frac(self._threshold))
        p.setPen(QPen(QColor(C["danger"]), 2))
        p.drawLine(QPointF(cx + (r_in - 4) * math.cos(ta), cy - (r_in - 4) * math.sin(ta)),
                   QPointF(cx + (r_out + 2) * math.cos(ta), cy - (r_out + 2) * math.sin(ta)))
        # Centre value + caption.
        p.setPen(QColor(C["ink"]))
        f = QFont(self.font())
        f.setPointSizeF(30)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.drawText(rect, int(Qt.AlignCenter), f"{self._value:.2f}")
        p.setPen(QColor(C["muted"]))
        fc = QFont(self.font())
        fc.setPointSizeF(10.5)
        p.setFont(fc)
        cap_rect = QRectF(rect.left(), cy + 18, rect.width(), 22)
        p.drawText(cap_rect, int(Qt.AlignHCenter | Qt.AlignTop), self._caption)
        p.end()


class Sparkline(QWidget):
    """A tiny trend line of recent drift values, coloured by current state."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(28)
        self._vals: list = []
        self._threshold = 0.65
        self._maxv = 2.0

    def set_values(self, vals, threshold: float) -> None:
        self._vals = [float(v) for v in (vals or [])][-40:]
        self._threshold = float(threshold)
        self.update()

    def apply_theme(self) -> None:
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if len(self._vals) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        pad = 3
        n = len(self._vals)
        mx = max(self._maxv, max(self._vals))
        def pt(i, v):
            x = pad + (w - 2 * pad) * (i / (n - 1))
            y = h - pad - (h - 2 * pad) * (v / mx)
            return QPointF(x, y)
        col = QColor(_drift_color(self._vals[-1], self._threshold))
        # Soft fill under the line.
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(pad, h - pad)
        for i, v in enumerate(self._vals):
            path.lineTo(pt(i, v))
        path.lineTo(w - pad, h - pad)
        path.closeSubpath()
        p.fillPath(path, _rgba(_drift_color(self._vals[-1], self._threshold), 28))
        # Line.
        pen = QPen(col, 1.8)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        for i in range(1, n):
            p.drawLine(pt(i - 1, self._vals[i - 1]), pt(i, self._vals[i]))
        # Last-point dot.
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(pt(n - 1, self._vals[-1]), 2.4, 2.4)
        p.end()


class StatCard(QFrame):
    """A small dashboard tile: icon chip + title, a big value, and a caption.

    Reference: the metric cards (Total Revenue / Active Contracts …). An optional
    extra widget (sparkline / gauge) can be docked under the value.
    """

    def __init__(self, icon: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("statCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(15, 13, 15, 13)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(8)
        chip = QLabel(icon)
        chip.setObjectName("iconChip")
        chip.setFixedSize(26, 26)
        chip.setAlignment(Qt.AlignCenter)
        ttl = QLabel(title)
        ttl.setObjectName("statTitle")
        head.addWidget(chip)
        head.addWidget(ttl)
        head.addStretch(1)
        self.trend = QLabel("")
        self.trend.setObjectName("statTrend")
        head.addWidget(self.trend)
        lay.addLayout(head)
        self.value = QLabel("—")
        self.value.setObjectName("statValue")
        lay.addWidget(self.value)
        self.caption = QLabel("")
        self.caption.setObjectName("statCaption")
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)
        self._extra_slot = lay

    def set_value(self, text: str) -> None:
        self.value.setText(text)

    def set_caption(self, text: str) -> None:
        self.caption.setText(text)

    def set_trend(self, text: str, good: bool = True) -> None:
        self.trend.setText(text)
        self.trend.setProperty("good", "1" if good else "0")
        self.trend.style().unpolish(self.trend)
        self.trend.style().polish(self.trend)

    def add_extra(self, widget: QWidget) -> None:
        self._extra_slot.addWidget(widget)


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


class SmartThread(QThread):
    """Run LLM-based smart drift analysis off the UI thread."""

    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, anchor, turns, provider, model) -> None:
        super().__init__()
        self._args = (anchor, turns, provider, model)

    def run(self) -> None:
        try:
            from cdm.smart import analyze

            anchor, turns, provider, model = self._args
            self.done.emit(analyze(anchor, turns, provider, model))
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
        self.model.setToolTip(
            "Aliases (opus / sonnet / haiku) always use the latest of that family.\n"
            "Pick a pinned ID (e.g. claude-opus-4-8) to lock an exact version."
        )
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

        self.status_lbl = QLabel()
        self.test_btn = QPushButton("Test connection")
        self.test_btn.clicked.connect(self._test_connection)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self.status_lbl, 1)
        status_row.addWidget(self.test_btn)

        self.model_caption = QLabel(
            "Pick a model from the list (or type one). ‘Refresh’ / ‘Test connection’ loads "
            "your provider's live list."
        )
        self.model_caption.setObjectName("muted")
        self.model_caption.setWordWrap(True)

        self.form = form
        self._key_field = self._wrap(key_row)
        form.addRow("Provider", self.provider)
        form.addRow("Status", self._wrap(status_row))
        form.addRow("Model", self._wrap(model_row))
        form.addRow(self.model_caption)   # full-width span: no mid-word wrap/overlap
        form.addRow("API key", self._key_field)
        form.addRow(self.hint)            # full-width span

        self.provider.currentIndexChanged.connect(lambda _i: self._load(self._current(), None))
        self.get_key_btn.clicked.connect(self._open_key_url)
        self.refresh_btn.clicked.connect(self._refresh_models)
        self._load(self._current(), model)

    def _set_status(self, ok: bool, text: str) -> None:
        color = "#1B7F3B" if ok else C["danger"]
        self.status_lbl.setText(f"<span style='color:{color}'>{'●' if ok else '○'}</span> {text}")

    def _test_connection(self) -> None:
        provider = self._current()
        if PROVIDERS[provider].get("keyless"):
            ready = claude_cli_available()
            self._set_status(ready, "Claude Code installed — ready (no key needed)"
                             if ready else "Claude Code not found — install it")
            return
        key = self.key.text().strip() or get_key(provider)
        if not key:
            self._set_status(False, "No API key entered")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ids = list_models(provider, key)
        except LLMError as exc:
            QApplication.restoreOverrideCursor()
            self._set_status(False, "Connection failed")
            QMessageBox.warning(self, "Drifter", str(exc))
            return
        QApplication.restoreOverrideCursor()
        current = self.model.currentText()
        self.model.clear()
        self.model.addItems(ids or curated_models(provider))
        self.model.setCurrentText(current if current in ids else (ids[0] if ids else current))
        self._set_status(True, f"Connected — {len(ids)} models loaded")

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
        # Hide the whole "API key" row for the keyless subscription (no empty field).
        try:
            self.form.setRowVisible(self._key_field, not keyless)
        except Exception:
            pass
        if keyless:
            self.model_caption.setText(
                "Aliases (opus / sonnet / haiku) always run the latest of that family. "
                "Pick a pinned ID like claude-opus-4-8 to lock an exact version — or type any."
            )
        else:
            self.model_caption.setText(
                "Pick a model from the list (or type one). ‘Refresh’ / ‘Test connection’ "
                "loads your provider's live list."
            )
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
        if keyless:
            ready = claude_cli_available()
            self._set_status(ready, "Claude Code installed — ready"
                             if ready else "Claude Code not found")
        else:
            ready = bool(get_key(provider))
            self._set_status(ready, "API key saved — connected"
                             if ready else "Not connected — add an API key")

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
        root.addWidget(logo_label(28))
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
            self.model = self.model or PROVIDERS[self.provider]["default_model"]
            self.monitor.store.set_meta("provider", self.provider)
            self.monitor.store.set_meta("model", self.model)
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
        self.resize(560, 660)
        self.setMinimumSize(500, 420)  # resizable + scrolls when short (responsive)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
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

            self.smart_check = QCheckBox("Smart analysis — let the LLM judge drift (recommended)")
            self.smart_check.setChecked((store.get_meta("smart") or "on") != "off")
            self.smart_check.setToolTip(
                "Uses your connected LLM to understand sub-goals and evolving goals, so "
                "deep work on part of a big goal isn't flagged as drift. Falls back to the "
                "offline engine when no LLM is connected."
            )
            lay.addWidget(self.smart_check)

            lay.addWidget(_hairline())
            capture = QLabel("Web chat capture")
            capture.setObjectName("h2")
            lay.addWidget(capture)
            cap_sub = QLabel(
                "Monitor a chat in another app (ChatGPT, Gemini…) by copying its replies — "
                "Drifter scores anything you copy against your goal. Starts a small "
                "background clipboard watcher; nothing leaves your Mac."
            )
            cap_sub.setObjectName("muted")
            cap_sub.setWordWrap(True)
            lay.addWidget(cap_sub)
            self.clip_check = QCheckBox("Capture clipboard")
            self.clip_check.setChecked(is_watcher_running())
            lay.addWidget(self.clip_check)

            lay.addWidget(_hairline())
            appearance = QLabel("Appearance")
            appearance.setObjectName("h2")
            lay.addWidget(appearance)
            self._theme_choice = store.get_meta("theme") or "auto"
            if self._theme_choice not in ("auto", "light", "dark"):
                self._theme_choice = "auto"
            seg = QHBoxLayout()
            seg.setSpacing(8)
            self._theme_btns = {}
            for key, label in (("auto", "Follow system"), ("light", "Light"), ("dark", "Dark")):
                btn = QPushButton(label)
                btn.clicked.connect(lambda _=False, k=key: self._set_theme(k))
                self._theme_btns[key] = btn
                seg.addWidget(btn)
            seg.addStretch(1)
            lay.addLayout(seg)
            self._sync_theme_seg()

        lay.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # Footer pinned below the scroll area, always visible.
        root.addWidget(_hairline())
        footer = QHBoxLayout()
        footer.setContentsMargins(28, 12, 28, 16)
        cancel = QPushButton("Cancel")
        save = QPushButton("Save")
        save.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self.accept)
        footer.addStretch(1)
        footer.addWidget(cancel)
        footer.addWidget(save)
        root.addLayout(footer)

    def _engine_pref(self) -> str:
        return (self.store.get_meta("embedder") if self.store else None) or config.EMBEDDER_PREFERENCE

    def _sync_engine_label(self) -> None:
        pref = self._engine_pref()
        if pref in ("semantic", "fastembed"):
            if fastembed_available():
                self.engine_state.setText(
                    "Current: Semantic (neural, runs offline after a one-time download). "
                    "Most accurate — tells reworded-but-on-topic from genuinely off-topic."
                )
            else:
                # Honest: selected semantic but the package isn't here, so it falls back.
                self.engine_state.setText(
                    "Selected: Semantic — but the ‘fastembed’ package isn’t installed, so "
                    "Drifter is actually running on Fast. Enable semantic below to fix it."
                )
        else:
            self.engine_state.setText(
                "Current: Fast (pure-Python, offline, zero download). It measures word "
                "overlap, so reworded-but-on-topic text can look like drift. Semantic is "
                "more accurate. Changes apply on restart."
            )

    def _use_fast(self) -> None:
        if self.store:
            self.store.set_meta("embedder", "hashing")
        self._sync_engine_label()
        QMessageBox.information(self, "Drifter", "Set to Fast. Restart Drifter to apply.")

    def _enable_semantic(self) -> None:
        if self._dl and self._dl.isRunning():
            return
        # Pre-check the package so we can give a precise fix instead of a raw stack trace.
        if not fastembed_available():
            QMessageBox.information(
                self, "Enable semantic drift",
                "Semantic mode needs one extra package, ‘fastembed’ (a small neural "
                "embedder — onnxruntime, no PyTorch).\n\n"
                "Install it, then click ‘Enable semantic’ again:\n\n"
                "    pip install fastembed\n\n"
                "or reinstall Drifter with the semantic extra:\n\n"
                "    pip install \"drifter[semantic]\"\n\n"
                "Until then Drifter keeps using the Fast offline engine.",
            )
            return
        prog = QProgressDialog(
            "Downloading the semantic model (~80 MB, one time)…\nIt runs fully offline afterwards.",
            None, 0, 0, self)
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
            low = error.lower()
            if "fastembed" in low and ("install" in low or "no module" in low or "requires" in low):
                msg = ("Semantic mode needs the ‘fastembed’ package. Install it and try "
                       "again:\n\n    pip install fastembed")
            else:
                msg = ("Couldn’t download the semantic model — this is almost always a "
                       "network/proxy issue. Check your internet connection and try again; "
                       "it downloads once (~80 MB), then runs fully offline.\n\n"
                       f"Details: {error}")
            QMessageBox.warning(self, "Couldn’t enable semantic", msg)
            return
        if self.store:
            self.store.set_meta("embedder", "semantic")
        self._sync_engine_label()
        QMessageBox.information(self, "Drifter", "Semantic drift enabled. Restart Drifter to apply.")

    def _set_theme(self, choice: str) -> None:
        """Apply an appearance choice live (restyles the whole app immediately)."""
        self._theme_choice = choice
        apply_theme_choice(choice, self.store)
        self._sync_theme_seg()

    def _sync_theme_seg(self) -> None:
        for key, btn in self._theme_btns.items():
            btn.setObjectName("segOn" if key == self._theme_choice else "seg")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

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

        # Optionally spin up a fresh Claude Code session seeded with this goal.
        self.cc_check = QCheckBox("Open a new Claude Code session in Terminal, seeded with this goal")
        self.cc_check.setChecked(claude_cli_available())
        self.cc_check.setEnabled(claude_cli_available())
        if not claude_cli_available():
            self.cc_check.setToolTip("Install Claude Code (the `claude` CLI) to enable this.")
        lay.addWidget(self.cc_check)
        dir_row = QHBoxLayout()
        folder_lbl = QLabel("Folder")
        folder_lbl.setObjectName("muted")
        self.cc_dir_edit = QLineEdit(str(Path.home()))
        self.cc_dir_edit.setToolTip("Working directory for the new Claude Code session")
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        dir_row.addWidget(folder_lbl)
        dir_row.addWidget(self.cc_dir_edit, 1)
        dir_row.addWidget(browse)
        lay.addLayout(dir_row)

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

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Working directory", self.cc_dir_edit.text())
        if d:
            self.cc_dir_edit.setText(d)

    def open_in_cc(self) -> bool:
        return self.cc_check.isChecked() and claude_cli_available()

    def cc_dir(self) -> str:
        return self.cc_dir_edit.text().strip() or str(Path.home())

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


def _cc_anchor(goal: str, constraints: List[str]) -> str:
    """System-prompt text that anchors a launched Claude Code session to the goal."""
    s = f"For this session, the user's goal is: {goal}."
    if constraints:
        s += " Hard constraints: " + "; ".join(constraints) + "."
        s += " Stay anchored to this goal and respect these constraints."
    return s


def _cc_kickoff(goal: str, constraints: List[str]) -> str:
    """First user message seeded into a launched Claude Code session."""
    s = f"My goal for this session: {goal}."
    if constraints:
        s += " Constraints: " + "; ".join(constraints) + "."
    return s + " Let's get started — how should we approach this?"


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
        root.addWidget(logo_label(26))
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
        root.addWidget(logo_label(28))
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
        if dlg.exec() != QDialog.Accepted:
            return
        name, goal, cons = dlg.values()
        if not goal:
            QMessageBox.warning(self, "Drifter", "A goal is required.")
            return
        session = self.monitor.start_session(name or "Untitled", goal, cons)
        if dlg.open_in_cc():
            self.chosen_tail_path = self._launch_and_attach_cc(
                dlg.cc_dir(), goal, cons, name or "Drifter session"
            )
        self.chosen_session_id = session.session_id
        self.accept()

    def _launch_and_attach_cc(self, cwd, goal, cons, name) -> Optional[str]:
        """Open a seeded Claude Code session in Terminal and wait for its transcript."""
        before = cc.snapshot_transcripts()
        ok = cc.launch_claude_in_terminal(
            cwd, kickoff=_cc_kickoff(goal, cons), anchor=_cc_anchor(goal, cons), name=name
        )
        if not ok:
            QMessageBox.warning(
                self, "Drifter",
                "Couldn't open a Terminal session. Start `claude` yourself, then use "
                "‘Monitor Claude Code…’ to attach.",
            )
            return None
        prog = QProgressDialog("Opening Claude Code — waiting for the session…", None, 0, 0, self)
        prog.setWindowTitle("Drifter")
        prog.setCancelButton(None)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()
        found = {"path": None}
        loop = QEventLoop()

        def _tick():
            p = cc.find_new_transcript(before, cwd)
            if p:
                found["path"] = p
                loop.quit()

        timer = QTimer(self)
        timer.setInterval(500)
        timer.timeout.connect(_tick)
        timer.start()
        QTimer.singleShot(15000, loop.quit)  # give up after ~15s
        loop.exec()
        timer.stop()
        prog.close()
        if not found["path"]:
            QMessageBox.information(
                self, "Drifter",
                "Opened Claude Code, but couldn't auto-detect the session yet. Once it's "
                "running, use ‘Monitor Claude Code…’ to attach it.",
            )
        return found["path"]

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
        if QMessageBox.question(
            self, "Remove session from Drifter",
            "Remove this session from Drifter?\n\nThis only deletes Drifter's local "
            "record (its drift history in Drifter's database). Your files, and your "
            "chat history with the LLM, are NOT touched.",
        ) == QMessageBox.StandardButton.Yes:
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
        provider = self.monitor.store.get_meta("provider") or first_connected_provider() or "claude"
        if provider not in PROVIDERS:
            provider = "claude"
        model = self.monitor.store.get_meta("model") or PROVIDERS[provider]["default_model"]
        dlg = SettingsDialog(provider, model, self.monitor.store, self)
        if dlg.exec() == QDialog.Accepted:
            p, m, _ = dlg.result_values()
            self.monitor.store.set_meta("provider", p)
            self.monitor.store.set_meta("model", m or PROVIDERS[p]["default_model"])

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
        saved_p = monitor.store.get_meta("provider")
        saved_m = monitor.store.get_meta("model")
        self.provider = saved_p if saved_p in PROVIDERS else default_provider()
        self.model = saved_m or PROVIDERS[self.provider]["default_model"]
        self._thread: Optional[ChatThread] = None
        self._stream_label: Optional[QLabel] = None
        self._stream_text = ""
        self._bubbles: List[QWidget] = []
        self.go_back = False  # set when the user returns to the session menu
        # Smart (LLM) analysis: on by default when a provider is connected.
        self.smart_enabled = (monitor.store.get_meta("smart") or "on") != "off"
        self.smart_verdict: Optional[dict] = None
        self._smart_thread: Optional[SmartThread] = None
        self._last_smart_n = -99
        # Tail mode: monitor a live Claude Code terminal transcript (read-only).
        self.tail = cc.ClaudeCodeTail(tail_path, start_at_end=True) if tail_path else None

        session = monitor.store.get_session(session_id)
        self.setWindowTitle(f"Drifter — {session.project_name if session else ''}")
        self.resize(1180, 768)
        self.setMinimumSize(820, 560)  # stays usable when shrunk
        self._build_ui(session)
        self._refresh_chart()
        self._update_coach()
        self._update_buttons()
        self._update_threshold_warning(self.monitor.threshold)

        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        if self not in _LIVE_WINDOWS:
            _LIVE_WINDOWS.append(self)  # so auto OS theme switches re-pen this window

    # -- layout -------------------------------------------------------------- #
    def _build_ui(self, session) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(logo_label(22))
        back_btn = QPushButton("‹ Sessions")
        back_btn.setObjectName("ghost")
        back_btn.setToolTip("Back to the session menu (your work is saved)")
        back_btn.clicked.connect(self._go_back)
        head.addWidget(back_btn)
        head.addWidget(_vsep())
        title = ElidingLabel(session.project_name if session else "Drifter")
        title.setObjectName("title")
        head.addWidget(title, 1)  # takes the slack; elides instead of overflowing
        self.provider_label = QLabel()
        self.provider_label.setObjectName("providerChip")
        head.addWidget(self.provider_label)
        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("iconBtn")
        settings_btn.setToolTip("Settings")
        settings_btn.setFixedSize(34, 34)
        settings_btn.clicked.connect(self._open_settings)
        head.addWidget(settings_btn)
        outer.addLayout(head)

        anchor = QLabel(f"Goal · {session.anchor_goal if session else ''}")
        anchor.setObjectName("anchor")
        anchor.setWordWrap(True)
        outer.addWidget(anchor)

        self.coach = QLabel("")
        self.coach.setObjectName("coach")
        self.coach.setWordWrap(True)
        outer.addWidget(self.coach)
        outer.addWidget(_hairline())

        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(10)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_chat_panel())
        split.addWidget(self._build_drift_panel())
        # Proportional resize: both panes grow/shrink with the window instead of one
        # pane eating all the extra space.
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([600, 560])
        outer.addWidget(split, 1)
        self._sync_provider_label()

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
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
        panel.setMinimumWidth(360)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(12)

        # Dashboard metric strip: a drift gauge tile + a status/forecast tile.
        lay.addWidget(self._build_metric_strip())

        chart_card = QFrame()
        chart_card.setObjectName("card")
        cc = QVBoxLayout(chart_card)
        cc.setContentsMargins(14, 12, 14, 12)
        cc.setSpacing(8)
        chead = QHBoxLayout()
        ctitle = QLabel("Drift over turns")
        ctitle.setObjectName("statTitle")
        reset_btn = QPushButton("⤢ Reset view")
        reset_btn.setObjectName("link")
        reset_btn.setToolTip("Reset zoom · scroll to zoom, drag to pan, hover for details")
        reset_btn.clicked.connect(lambda: self.chart.reset_view())
        chead.addWidget(ctitle)
        chead.addStretch(1)
        chead.addWidget(reset_btn)
        cc.addLayout(chead)
        self.chart = DriftChart()
        self.chart.setMinimumHeight(190)  # shrinks with the window, stays legible
        cc.addWidget(self.chart)
        cc.addWidget(_hairline())
        cc.addWidget(self._build_legend_strip())  # small, always-visible key inside the card
        _shadow(chart_card)
        lay.addWidget(chart_card, 1)

        # One tidy control bar: threshold + auto re-align live here. Everything
        # explanatory lives in the legend (above) and the "?" help dialog.
        bar = QFrame()
        bar.setObjectName("toolbar")
        th_row = QHBoxLayout(bar)
        th_row.setContentsMargins(14, 10, 12, 10)
        th_row.setSpacing(8)
        th_lbl = QLabel("Threshold")
        th_lbl.setObjectName("muted")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.30, 0.95)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(round(self.monitor.threshold, 2))
        self.threshold_spin.setFixedWidth(86)
        self.threshold_spin.setToolTip(
            "How far the conversation may drift from your goal before Drifter warns you.\n"
            "Lower = stricter. Drift: 0 = on goal, 1 = unrelated."
        )
        self.threshold_spin.valueChanged.connect(self._on_threshold)
        help_btn = QPushButton("?")
        help_btn.setObjectName("link")
        help_btn.setFixedWidth(24)
        help_btn.setToolTip("What does the threshold mean?")
        help_btn.clicked.connect(self._explain_threshold)
        self.auto_check = QCheckBox("Auto re-align")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip("Fold the corrective prompt into the next reply automatically when drift fires.")
        th_row.addWidget(th_lbl)
        th_row.addWidget(self.threshold_spin)
        th_row.addWidget(help_btn)
        th_row.addStretch(1)
        th_row.addWidget(self.auto_check)
        lay.addWidget(bar)

        # Extreme-threshold caption — subtle inline hint, only when at an extreme.
        self.threshold_warn = QLabel("")
        self.threshold_warn.setObjectName("warn")
        self.threshold_warn.setWordWrap(True)
        self.threshold_warn.setVisible(False)
        lay.addWidget(self.threshold_warn)

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

    def _build_metric_strip(self) -> QWidget:
        """A dashboard metric strip: a drift gauge tile + a status/forecast tile."""
        strip = QWidget()
        row = QHBoxLayout(strip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        # -- gauge tile -- #
        gcard = QFrame()
        gcard.setObjectName("statCard")
        gv = QVBoxLayout(gcard)
        gv.setContentsMargins(12, 11, 12, 8)
        gv.setSpacing(2)
        gh = QHBoxLayout()
        gh.setSpacing(8)
        gi = QLabel("◎")
        gi.setObjectName("iconChip")
        gi.setFixedSize(26, 26)
        gi.setAlignment(Qt.AlignCenter)
        gt = QLabel("Drift")
        gt.setObjectName("statTitle")
        self.gauge_turns = QLabel("0 turns")
        self.gauge_turns.setObjectName("statCaption")
        gh.addWidget(gi)
        gh.addWidget(gt)
        gh.addStretch(1)
        gh.addWidget(self.gauge_turns)
        gv.addLayout(gh)
        self.gauge = DriftGauge()
        gv.addWidget(self.gauge, 1)
        _shadow(gcard, blur=22, dy=5, alpha=18)
        row.addWidget(gcard, 3)

        # -- status / forecast tile -- #
        scard = QFrame()
        scard.setObjectName("statCard")
        sv = QVBoxLayout(scard)
        sv.setContentsMargins(14, 11, 14, 12)
        sv.setSpacing(7)
        sh = QHBoxLayout()
        sh.setSpacing(8)
        si = QLabel("◉")
        si.setObjectName("iconChip")
        si.setFixedSize(26, 26)
        si.setAlignment(Qt.AlignCenter)
        st = QLabel("Status")
        st.setObjectName("statTitle")
        self.status_pill = QLabel("on track")
        self.status_pill.setObjectName("pillOk")
        sh.addWidget(si)
        sh.addWidget(st)
        sh.addStretch(1)
        sh.addWidget(self.status_pill)
        sv.addLayout(sh)
        self.status_reason = QLabel("Monitoring — keep chatting.")
        self.status_reason.setObjectName("statCaption")
        self.status_reason.setWordWrap(True)
        sv.addWidget(self.status_reason)
        self.spark = Sparkline()
        self.spark.setFixedHeight(32)
        sv.addWidget(self.spark)
        sv.addWidget(_hairline())
        self.fc_label = QLabel("Forecast: stable")
        self.fc_label.setObjectName("statCaption")
        self.fc_label.setWordWrap(True)
        sv.addWidget(self.fc_label)
        _shadow(scard, blur=22, dy=5, alpha=18)
        row.addWidget(scard, 4)
        return strip

    def _set_status(self, kind: str, pill_text: str, reason: str) -> None:
        """Set the status tile's pill (pillOk/pillBad/pillWarn) + reason line."""
        self.status_pill.setObjectName(kind)
        self.status_pill.setText(pill_text)
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)
        self.status_reason.setText(reason)

    def _restyle_glance(self) -> None:
        """Repaint the gauge + sparkline for the active palette (theme change)."""
        if hasattr(self, "gauge"):
            self.gauge.apply_theme()
        if hasattr(self, "spark"):
            self.spark.apply_theme()

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
    # Short, header-friendly provider names (the full labels are too long for a chip).
    _PROVIDER_SHORT = {
        "claude-cli": "Claude", "claude": "Claude", "gemini": "Gemini", "openai": "OpenAI",
    }

    def _sync_provider_label(self) -> None:
        ready = provider_ready(self.provider)
        dot = "●" if ready else "○"
        name = self._PROVIDER_SHORT.get(self.provider, PROVIDERS[self.provider]["label"])
        self.provider_label.setText(f"{dot} {name} · {self.model}")
        self.provider_label.setToolTip(
            f"{PROVIDERS[self.provider]['label']} · {self.model}"
            + ("" if ready else "  (not connected — open Settings)")
        )

    def _update_coach(self) -> None:
        if self.smart_verdict:
            v = self.smart_verdict
            if v["status"] == "drifting":
                self.coach.setText("⚠️ Smart: off-track — " + v.get("reason", "") +
                                   " Use the corrective on the right.")
            else:
                nice = {"on_track": "on track", "sub_task": "on track — working on a sub-task of your goal",
                        "evolved": "your goal evolved — anchor updated"}.get(v["status"], "on track")
                self.coach.setText(f"🧠 Smart: {nice} — {v.get('reason', '')}")
            return
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
    def _go_back(self) -> None:
        """Return to the session menu without quitting the app (work is saved)."""
        self.go_back = True
        self.close()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.provider, self.model, self.monitor.store, self)
        accepted = dlg.exec() == QDialog.Accepted
        # Appearance can change live inside the dialog — re-tint chart + legend + tiles either way.
        self.chart.apply_theme()
        self._restyle_legend()
        self._restyle_glance()
        if accepted:
            provider, model, _ = dlg.result_values()
            self.provider = provider
            self.model = model or PROVIDERS[provider]["default_model"]
            self.monitor.store.set_meta("provider", self.provider)
            self.monitor.store.set_meta("model", self.model)
            if hasattr(dlg, "smart_check"):
                on = dlg.smart_check.isChecked()
                self.monitor.store.set_meta("smart", "on" if on else "off")
                self.smart_enabled = on
                if not on:
                    self.smart_verdict = None
            # Clipboard capture lives in Settings now — apply on save (only if changed,
            # so re-saving doesn't spawn a duplicate watcher).
            if hasattr(dlg, "clip_check"):
                want = dlg.clip_check.isChecked()
                if want != is_watcher_running():
                    self._on_clip_toggle(want)
            self._sync_provider_label()
            self._update_coach()
            self._refresh_chart()

    def _explain_threshold(self) -> None:
        QMessageBox.information(
            self, "Reading the drift chart",
            "Drift is how far the conversation has moved from your original goal — "
            "0 means right on it, 1 means completely unrelated.\n\n"
            "The chart shows two lines:\n"
            "• Goal (bold orange) — drift from your original goal.\n"
            "• Recent context (grey dashed) — drift from the recent conversation, i.e. "
            "whether you're also wandering turn-to-turn.\n\n"
            "The threshold (red dotted line) is how much drift you'll tolerate before "
            "Drifter warns you and offers a corrective prompt. Lower = stricter. "
            "Defaults: 0.65 in semantic mode, 0.80 in fast offline mode (its numbers run "
            "higher).\n\n"
            "The shaded band (‘Normal range’) is the normal spread learned from the start "
            "of THIS conversation, so a rise above the band is a genuine shift rather than "
            "noise. The dashed projection forecasts when drift will cross the threshold.",
        )

    def _update_threshold_warning(self, value: float) -> None:
        if value <= 0.42:
            self.threshold_warn.setText(
                "Very strict — almost any tangent counts as drift. This can force the "
                "conversation into a very niche slice of your goal."
            )
            self.threshold_warn.setVisible(True)
        elif value >= 0.90:
            self.threshold_warn.setText(
                "Very loose — drift will rarely be flagged, so off-track turns may slip by."
            )
            self.threshold_warn.setVisible(True)
        else:
            self.threshold_warn.setVisible(False)

    def _build_legend_strip(self) -> QFrame:
        """A small, always-visible key docked inside the chart card.

        Four atomic [painted-swatch + label] items in a 2-column grid so whole
        entries reflow on narrow widths instead of word-wrapping through an entry.
        Swatches are painted from the chart's exact pens (see :func:`_legend_swatch`).
        """
        strip = QFrame()
        strip.setObjectName("legendStrip")
        grid = QGridLayout(strip)
        grid.setContentsMargins(2, 8, 2, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)
        self._legend_items: List = []
        entries = [("goal", "Goal"), ("recent", "Recent context"),
                   ("threshold", "Threshold"), ("band", "Normal range")]
        dpr = self.devicePixelRatioF() or 2.0
        for i, (kind, text) in enumerate(entries):
            item = QWidget()
            row = QHBoxLayout(item)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            sw = QLabel()
            sw.setObjectName("legendSwatch")
            sw.setFixedSize(28, 14)
            sw.setPixmap(_legend_swatch(kind, dpr))
            sw.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            key = QLabel(text)
            key.setObjectName("legendKey")
            row.addWidget(sw)
            row.addWidget(key)
            row.addStretch(1)
            grid.addWidget(item, i // 2, i % 2)
            self._legend_items.append((sw, kind))
        return strip

    def _restyle_legend(self) -> None:
        """Repaint legend swatches for the current palette (call on theme change)."""
        if not hasattr(self, "_legend_items"):
            return
        dpr = self.devicePixelRatioF() or 2.0
        for sw, kind in self._legend_items:
            sw.setPixmap(_legend_swatch(kind, dpr))

    def _on_threshold(self, value: float) -> None:
        self.monitor.set_threshold(float(value))
        self._update_threshold_warning(value)
        self._refresh_chart()  # re-renders the (now threshold-aware) corrective

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
    def _maybe_smart(self) -> None:
        """Kick off LLM analysis every few turns when Smart mode is available."""
        if not self.smart_enabled or not provider_ready(self.provider):
            return
        if self._smart_thread and self._smart_thread.isRunning():
            return
        msgs = self.monitor.store.get_messages(self.session_id)
        if len(msgs) < 2 or (len(msgs) - self._last_smart_n) < 3:
            return
        self._last_smart_n = len(msgs)
        session = self.monitor.store.get_session(self.session_id)
        if session is None:
            return
        turns = [{"role": m.role, "text": m.text} for m in msgs]
        self._smart_thread = SmartThread(session.anchor_goal, turns, self.provider, self.model)
        self._smart_thread.done.connect(self._on_smart_done)
        self._smart_thread.failed.connect(self._on_smart_failed)
        self._smart_thread.start()

    def _on_smart_done(self, verdict: dict) -> None:
        self.smart_verdict = verdict
        self._update_smart_ui()
        self._update_coach()

    def _on_smart_failed(self, _msg: str) -> None:
        # Stay on the offline signal; permit a retry on the next cadence.
        self._last_smart_n = -99

    def _update_smart_ui(self) -> None:
        """Smart verdict (when present) is authoritative over the offline status tile.

        Drives the status pill + reason and the corrective card. The verdict reason is
        also surfaced in the coach bar (see ``_update_coach``).
        """
        v = self.smart_verdict
        if not v:
            return
        reason = v.get("reason", "")
        if v["status"] == "drifting":
            self._set_status("pillBad", "DRIFTING", reason or "Off your goal.")
            base = v.get("corrective")
            if base:  # LLM-written corrective — add the threshold-tuned tightness line
                corr = base + "\n\n" + strictness_line(self.monitor.threshold)
            else:  # offline corrective is already threshold-aware
                corr = self.monitor.current_corrective_prompt(self.session_id)
            self.corr_text.setPlainText(corr)
            self.corr_card.setVisible(True)
        else:
            pill, label = {
                "on_track": ("pillOk", "on track"),
                "sub_task": ("pillWarn", "sub-task"),
                "evolved": ("pillWarn", "goal evolved"),
            }.get(v["status"], ("pillOk", "on track"))
            self._set_status(pill, label, reason or "On your goal.")
            self.corr_card.setVisible(False)

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
        self._maybe_smart()

    def _refresh_chart(self) -> None:
        try:
            ts = self.monitor.timeseries(self.session_id)
        except Exception:
            return
        turns = ts.get("turns") or []
        anchor = ts.get("drift_from_anchor") or []
        threshold = float(ts.get("threshold", self.monitor.threshold))
        self.chart.update(ts)
        last = anchor[-1] if anchor else 0.0
        high = bool(turns) and (last > threshold)

        # Drive the dashboard metric tiles.
        self.gauge.set_state(last, threshold, "DRIFTING" if high else "on track")
        self.gauge_turns.setText(f"{len(turns)} turn{'' if len(turns) == 1 else 's'}")
        self.spark.set_values(anchor, threshold)
        self._set_forecast_label(ts, high)
        if high:
            self._set_status("pillBad", "DRIFTING", "Off your goal — review the corrective.")
            self.corr_text.setPlainText(self.monitor.current_corrective_prompt(self.session_id))
            self.corr_card.setVisible(True)
        else:
            self._set_status("pillOk", "on track", "Monitoring — keep chatting.")
            self.corr_card.setVisible(False)
        self._update_smart_ui()  # smart verdict overrides the offline status when present

    def _set_forecast_label(self, ts: dict, high: bool) -> None:
        """Forecast tile copy from the self-calibrating analytics in ``ts``."""
        if high:
            self.fc_label.setText("⚠ Off-track now")
            return
        will = ts.get("forecast_will_cross")
        fc = ts.get("forecast_turns")
        if will and fc:
            self.fc_label.setText(f"↗ Forecast: crosses in ~{int(fc)} turns")
        else:
            self.fc_label.setText("Forecast: stable")

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            if self in _LIVE_WINDOWS:
                _LIVE_WINDOWS.remove(self)
            self._timer.stop()
            if self._thread and self._thread.isRunning():
                self._thread.stop()
                self._thread.wait(2000)
            if self._smart_thread and self._smart_thread.isRunning():
                self._smart_thread.wait(50)
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

    from cdm.storage import Store

    store = Store()

    # Theme: honour the user's saved choice; otherwise follow macOS.
    theme = store.get_meta("theme")
    if theme == "dark":
        set_dark(True)
    elif theme == "light":
        set_dark(False)
    else:
        try:
            set_dark(app.styleHints().colorScheme() == Qt.ColorScheme.Dark)
        except Exception:
            set_dark(False)
    app.setStyleSheet(build_qss())

    def _on_scheme(_s=None):
        if store.get_meta("theme") in (None, "auto"):  # only auto-follow if unset
            try:
                set_dark(app.styleHints().colorScheme() == Qt.ColorScheme.Dark)
            except Exception:
                return
            app.setStyleSheet(build_qss())
            # QSS doesn't repaint pyqtgraph pens or painted swatches — do it explicitly.
            for w in list(_LIVE_WINDOWS):
                try:
                    w.chart.apply_theme()
                    w._restyle_legend()
                    w._restyle_glance()
                except Exception:
                    pass

    try:
        app.styleHints().colorSchemeChanged.connect(_on_scheme)
    except Exception:
        pass
    mono = QFont()
    mono.setFamilies(["SF Mono", "JetBrains Mono", "Menlo", "Cascadia Code", "Consolas", "monospace"])
    mono.setPointSize(13)
    mono.setStyleHint(QFont.Monospace)
    app.setFont(mono)
    app.setWindowIcon(QIcon(_asset("drifter_icon.png")))

    show_splash(app)  # logo, fades in then out

    pref = store.get_meta("embedder") or config.EMBEDDER_PREFERENCE
    monitor = DriftMonitor(store=store, embedder=safe_embedder(pref))

    # Loop: menu -> session window -> (Back) -> menu … until the user quits.
    while True:
        first_run = not load_profile_name() and not monitor.store.list_sessions()
        dlg: QDialog = OnboardingWizard(monitor) if first_run else LaunchDialog(monitor)
        if dlg.exec() != QDialog.Accepted or not dlg.chosen_session_id:
            break
        monitor.store.set_active_session(dlg.chosen_session_id)
        window = MainWindow(
            monitor, dlg.chosen_session_id, tail_path=getattr(dlg, "chosen_tail_path", None)
        )
        window.show()
        app.exec()  # returns when the window closes
        if not window.go_back:
            break  # closed via the window button / quit -> exit the app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

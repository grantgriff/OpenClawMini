"""Retro orange gradient terminal styling for OpenClawMini."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

console = Console()

COLORS = {
    "orange_1": "#FF4500",
    "orange_2": "#FF6B35",
    "orange_3": "#FF8C42",
    "orange_4": "#FFA54F",
    "orange_5": "#FFB347",
    "orange_6": "#FFC966",
    "orange_7": "#FFD93D",
    "bg": "#1A1A2E",
    "text": "#EAEAEA",
}

# ASCII art lines paired with gradient colors (top=dark, bottom=bright)
_BANNER_LINES = [
    ("#FF4500", " ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗      █████╗ ██╗    ██╗"),
    ("#FF5A1F", " ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║     ██╔══██╗██║    ██║"),
    ("#FF6B35", " ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ███████║██║ █╗ ██║"),
    ("#FF8C42", " ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██╔══██║██║███╗██║"),
    ("#FFA54F", " ╚██████╔╝██║     ███████╗██║  ╚████║╚██████╗███████╗██║  ██║╚███╔███╔╝"),
    ("#FFB347", "  ╚═════╝ ╚═╝     ╚══════╝╚═╝   ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝"),
    ("", ""),
    ("#FF6B35", " ███╗   ███╗██╗███╗   ██╗██╗"),
    ("#FF8C42", " ████╗ ████║██║████╗  ██║██║"),
    ("#FFA54F", " ██╔████╔██║██║██╔██╗ ██║██║"),
    ("#FFB347", " ██║╚██╔╝██║██║██║╚██╗██║██║"),
    ("#FFC966", " ██║  ╚═╝ ██║██║██║  ╚████║██║"),
    ("#FFD93D", " ╚═╝      ╚═╝╚═╝╚═╝   ╚═══╝╚═╝"),
]


def print_banner() -> None:
    """Print the OpenClawMini ASCII banner with retro orange gradient."""
    console.print()
    for color, line in _BANNER_LINES:
        if color:
            console.print(f"[bold {color}]{line}[/]")
        else:
            console.print()
    console.print()
    console.print(
        f"[{COLORS['orange_2']}] 🟠🟠🟠[/] "
        f"[bold {COLORS['orange_5']}]Your Personal AI That Learns To Be You[/] "
        f"[{COLORS['orange_2']}]🟠🟠🟠[/]"
    )
    console.print()


def print_panel(message: str, title: str = "", style: str = "orange1") -> None:
    """Print a styled Rich panel."""
    console.print(Panel(message, title=title, border_style=style))


def print_success(message: str) -> None:
    console.print(f"[bold {COLORS['orange_5']}]✓[/] {message}")


def print_step(step_num: int, title: str) -> None:
    console.print(f"\n[bold {COLORS['orange_2']}]Step {step_num}:[/] [bold]{title}[/]")
    console.print(f"[{COLORS['orange_4']}]{'─' * 40}[/]")


def make_progress() -> Progress:
    """Return a Rich Progress bar with orange spinner."""
    return Progress(
        SpinnerColumn(style=COLORS["orange_2"]),
        TextColumn(f"[bold {COLORS['orange_3']}]{{task.description}}[/]"),
        BarColumn(bar_width=40, style=COLORS["orange_4"], complete_style=COLORS["orange_2"]),
        TaskProgressColumn(),
        console=console,
    )


def print_progress_table(results: list) -> None:
    """Print a training progress table from a list of EvalResults-like dicts."""
    table = Table(
        title=f"[bold {COLORS['orange_2']}]📊 Training Progress[/]",
        show_header=True,
        header_style=f"bold {COLORS['orange_2']}",
        border_style=COLORS["orange_4"],
    )
    table.add_column("Stage", style="cyan", min_width=8)
    table.add_column("Factual", justify="right", min_width=8)
    table.add_column("Stylistic", justify="right", min_width=9)
    table.add_column("Overall", justify="right", min_width=8)
    table.add_column("Δ", justify="right", style="green", min_width=6)

    prev_overall = 0.0
    for r in results:
        overall = r.get("overall_accuracy", 0.0)
        delta = overall - prev_overall
        delta_str = (f"[green]+{delta:.0%}[/]" if delta > 0 else f"[red]{delta:.0%}[/]") if prev_overall > 0 else "-"
        table.add_row(
            r.get("stage", "?").upper(),
            f"{r.get('factual_accuracy', 0):.0%}",
            f"{r.get('stylistic_accuracy', 0):.0%}",
            f"[bold]{overall:.0%}[/]",
            delta_str,
        )
        prev_overall = overall

    console.print(table)

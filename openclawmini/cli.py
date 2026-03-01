"""OpenClawMini CLI - Typer app with retro orange terminal styling."""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from openclawmini.config import (
    DATA_GEN_OPTIONS,
    ORCHESTRATOR_OPTIONS,
    RESEARCH_OPTIONS,
    Config,
    TrainingConfig,
    UserConfig,
    load_config,
    load_env,
    save_config,
    save_env,
)
from openclawmini.styling import (
    COLORS,
    console,
    print_banner,
    print_panel,
    print_step,
    print_success,
    print_progress_table,
)

app = typer.Typer(
    name="openclawmini",
    help="Your Personal AI That Learns To Be You.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _divider() -> None:
    console.print(f"[{COLORS['orange_4']}]{'═' * 72}[/]")


def _prompt_model_choice(title: str, options: list[dict]) -> dict:
    """Display a numbered model-selection menu and return the chosen option."""
    console.print(f"\n[bold {COLORS['orange_2']}]{title}:[/]")
    for i, opt in enumerate(options, 1):
        default_tag = " [dim]← Default[/]" if opt.get("default") else ""
        console.print(f"  [{COLORS['orange_4']}][{i}][/] {opt['label']}{default_tag}")

    while True:
        raw = Prompt.ask(
            f"  [dim]Choice[/]",
            default="1",
            console=console,
        )
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = Prompt.ask(prompt, default=str(default), console=console)
        try:
            return float(raw)
        except ValueError:
            console.print("  [red]Please enter a valid number[/]")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = Prompt.ask(prompt, default=str(default), console=console)
        try:
            return int(raw)
        except ValueError:
            console.print("  [red]Please enter a whole number[/]")


# ─────────────────────────────────────────────────────────────
# OAuth stubs
# ─────────────────────────────────────────────────────────────

def _setup_gmail_oauth(env_vars: dict) -> dict:
    """Walk user through Gmail OAuth setup (stub - real auth in Task 3)."""
    console.print(f"\n[{COLORS['orange_3']}]To access your sent emails, OpenClawMini needs read-only Gmail access.[/]")
    console.print(f"[dim]Scope: https://www.googleapis.com/auth/gmail.readonly[/]")
    console.print(f"[dim]This only allows READING emails — no send/delete/modify permissions.[/]\n")

    client_id = Prompt.ask(
        f"  [bold]Gmail Client ID[/] [dim](from Google Cloud Console)[/]",
        default="",
        console=console,
    )
    client_secret = Prompt.ask(
        f"  [bold]Gmail Client Secret[/]",
        default="",
        password=True,
        console=console,
    )

    if client_id and client_secret:
        env_vars["GMAIL_CLIENT_ID"] = client_id
        env_vars["GMAIL_CLIENT_SECRET"] = client_secret
        env_vars["GMAIL_REDIRECT_URI"] = "http://localhost:8080/callback"
        env_vars["GMAIL_TOKEN_PATH"] = "./data/gmail_token.json"

        console.print(f"\n[{COLORS['orange_4']}]OAuth flow will run when you execute [bold]openclawmini run[/].[/]")
        console.print(f"[dim]Your browser will open to authorize Gmail access at that time.[/]")
        print_success("Gmail credentials saved.")
    else:
        console.print(f"[dim]Skipping Gmail — you can add credentials later in .env[/]")

    return env_vars


def _setup_linkedin_oauth(env_vars: dict) -> dict:
    """Walk user through LinkedIn OAuth setup (stub - real auth in Task 3)."""
    console.print(f"\n[{COLORS['orange_3']}]To access your LinkedIn profile and posts, we need read-only API access.[/]")
    console.print(f"[dim]Scopes: r_liteprofile, r_emailaddress, r_member_social (read only — no posting)[/]\n")

    client_id = Prompt.ask(
        f"  [bold]LinkedIn Client ID[/] [dim](from LinkedIn Developer Portal)[/]",
        default="",
        console=console,
    )
    client_secret = Prompt.ask(
        f"  [bold]LinkedIn Client Secret[/]",
        default="",
        password=True,
        console=console,
    )

    if client_id and client_secret:
        env_vars["LINKEDIN_CLIENT_ID"] = client_id
        env_vars["LINKEDIN_CLIENT_SECRET"] = client_secret
        console.print(f"\n[{COLORS['orange_4']}]LinkedIn OAuth flow will run when you execute [bold]openclawmini run[/].[/]")
        print_success("LinkedIn credentials saved.")
    else:
        console.print(f"[dim]Skipping LinkedIn — you can add credentials later in .env[/]")

    return env_vars


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────

@app.command("init")
def cmd_init() -> None:
    """Interactive first-run setup: connect data sources, configure models."""
    print_banner()
    print_panel(
        f"[bold {COLORS['orange_5']}]Welcome! Let's get you set up.[/]",
        title="🟠 OpenClawMini Setup",
    )

    config = Config()
    env_vars: dict[str, str] = {}

    # ── Step 1: User info ──────────────────────────────────────
    print_step(1, "User Information")
    config.user.name = Prompt.ask("  [bold]Full name[/]", console=console)
    config.user.email = Prompt.ask("  [bold]Email address[/]", console=console)

    # ── Step 2: Data sources ───────────────────────────────────
    print_step(2, "Data Sources")
    console.print(f"  [{COLORS['orange_4']}]Which sources should we use to learn about you?[/]\n")

    use_gmail = Confirm.ask("  [bold]Gmail[/] (sent emails for writing samples)", default=True, console=console)
    use_linkedin = Confirm.ask("  [bold]LinkedIn[/] (profile and posts)", default=True, console=console)
    use_web = Confirm.ask("  [bold]Web search[/] (public mentions)", default=True, console=console)
    use_files = Confirm.ask(
        "  [bold]File upload[/] (ChatGPT/Claude exports, conversation logs)",
        default=False,
        console=console,
    )

    config.user.data_sources = {
        "gmail": use_gmail,
        "linkedin": use_linkedin,
        "web_search": use_web,
        "file_upload": use_files,
    }

    # ── Step 3: Gmail OAuth ────────────────────────────────────
    if use_gmail:
        print_step(3, "Connect Gmail")
        env_vars = _setup_gmail_oauth(env_vars)
    else:
        console.print(f"\n[dim]Skipping Gmail setup.[/]")

    # ── Step 4: LinkedIn ───────────────────────────────────────
    if use_linkedin:
        print_step(4, "Connect LinkedIn")
        console.print(
            f"\n[{COLORS['orange_3']}]LinkedIn public profile scraping (no API key needed).[/]"
        )
        console.print(
            f"[dim]We'll scrape your public LinkedIn profile for professional facts.[/]\n"
        )
        linkedin_url = Prompt.ask(
            f"  [bold]LinkedIn profile URL[/] [dim](e.g. https://linkedin.com/in/yourname)[/]",
            default="",
            console=console,
        )
        if linkedin_url.strip():
            env_vars["LINKEDIN_PROFILE_URL"] = linkedin_url.strip()
            console.print(f"  [dim]LinkedIn URL saved.[/]")
        else:
            console.print(f"  [dim]Skipping — you can add LINKEDIN_PROFILE_URL to .env later.[/]")

        # Also offer full API OAuth setup
        want_oauth = Confirm.ask(
            "  Set up LinkedIn API OAuth (for posts/deeper data)?",
            default=False,
            console=console,
        )
        if want_oauth:
            env_vars = _setup_linkedin_oauth(env_vars)
    else:
        console.print(f"\n[dim]Skipping LinkedIn setup.[/]")

    # ── Step 5: Training config ────────────────────────────────
    print_step(5, "Training Configuration")
    _divider()
    config.training.sft_sample_count = _prompt_int(
        f"  [bold]SFT training samples[/] [dim][default: 200][/]", 200
    )
    config.training.grpo_scenario_count = _prompt_int(
        f"  [bold]GRPO scenarios[/] [dim][default: 100][/]", 100
    )
    config.training.quality_threshold = _prompt_float(
        f"  [bold]Quality threshold (1-10)[/] [dim][default: 7.0][/]", 7.0
    )
    config.training.sft_factual_threshold = _prompt_float(
        f"  [bold]Target factual accuracy for SFT → GRPO handoff[/] [dim][default: 0.70][/]", 0.70
    )
    config.training.final_target_accuracy = _prompt_float(
        f"  [bold]Final target persona accuracy[/] [dim][default: 0.80][/]", 0.80
    )

    # ── Step 6: Model configuration ───────────────────────────
    print_step(6, "Model Configuration")
    _divider()

    use_defaults = Confirm.ask("  Use default model configuration?", default=True, console=console)

    if not use_defaults:
        orch = _prompt_model_choice("ORCHESTRATOR AGENT (Autonomous reasoning)", ORCHESTRATOR_OPTIONS)
        research = _prompt_model_choice("RESEARCH EXTRACTION (Fact extraction)", RESEARCH_OPTIONS)
        datagen = _prompt_model_choice("DATA GENERATION (Training data creation)", DATA_GEN_OPTIONS)

        from openclawmini.config import ModelConfig
        config.orchestrator = ModelConfig(orch["provider"], orch["model"])
        config.research_extraction = ModelConfig(research["provider"], research["model"])
        config.data_generation = ModelConfig(datagen["provider"], datagen["model"])

    # Show summary
    console.print(f"\n[bold {COLORS['orange_3']}]Model Summary:[/]")
    console.print(f"  Orchestrator:  [cyan]{config.orchestrator.model}[/]")
    console.print(f"  Research:      [cyan]{config.research_extraction.model}[/]")
    console.print(f"  Data Gen:      [cyan]{config.data_generation.model}[/]")
    console.print(f"  GRPO Judge:    [cyan]{config.grpo_judge.model}[/]")
    console.print(f"  Eval Judge:    [cyan]{config.eval_judge.model}[/]")
    console.print(f"  Base Model:    [cyan]{config.base_model.model}[/]")

    # ── API keys ───────────────────────────────────────────────
    console.print(f"\n[bold {COLORS['orange_3']}]API Keys:[/]")
    console.print(f"  [{COLORS['orange_4']}]Enter your API keys (press Enter to skip):[/]\n")

    for key_name, label in [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("GOOGLE_API_KEY", "Google (Gemini)"),
        ("MISTRAL_API_KEY", "Mistral (REQUIRED for fine-tuning)"),
        ("WANDB_API_KEY", "W&B (REQUIRED for logging)"),
        ("OPENAI_API_KEY", "OpenAI (optional)"),
    ]:
        existing = os.getenv(key_name, "")
        if existing:
            console.print(f"  [dim]{label}: already set ✓[/]")
        else:
            val = Prompt.ask(f"  [bold]{label}[/]", default="", password=True, console=console)
            if val:
                env_vars[key_name] = val

    for key_name, label, default in [
        ("WANDB_ENTITY", "W&B entity/username", ""),
        ("WANDB_PROJECT", "W&B project name", "openclawmini"),
    ]:
        val = Prompt.ask(f"  [bold]{label}[/]", default=default, console=console)
        if val:
            env_vars[key_name] = val

    # ── Write files ────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    save_config(config, "config.yaml")
    save_env(env_vars, ".env")

    console.print()
    print_panel(
        f"[bold {COLORS['orange_5']}]✓ Setup complete![/]\n\n"
        f"  [bold cyan]openclawmini train[/]  — autonomous: collects memory + loops SFT→GRPO until done\n"
        f"  [bold cyan]openclawmini run[/]    — manual: step-by-step wizard, one pass through the pipeline",
        title="🟠 OpenClawMini",
    )


@app.command("run")
def cmd_run() -> None:
    """Manual pipeline wizard — step-by-step, one pass through research → SFT → GRPO.

    Use 'openclawmini train' for the fully autonomous loop that dynamically
    repeats SFT/GRPO until your target accuracy is reached.
    """
    load_env()
    print_banner()

    memory_path = Path("data/memory.json")
    config_path = Path("config.yaml")

    if not config_path.exists():
        print_panel(
            f"[red]No config.yaml found.[/]\n\nRun [bold cyan]openclawmini init[/] first.",
            title="Error",
            style="red",
        )
        raise typer.Exit(1)

    config = load_config("config.yaml")

    if memory_path.exists():
        # Continuing session — load and display memory stats
        from openclawmini.memory import MemoryStore
        store = MemoryStore(str(memory_path))
        memory = store.load()
        stats = memory.stats()
        last_updated = memory.user.last_updated.strftime("%Y-%m-%d %H:%M") if memory.user.last_updated else "unknown"

        stats_lines = (
            f"  Facts:           [cyan]{stats['facts']}[/]\n"
            f"  Writing samples: [cyan]{stats['writing_samples']}[/]\n"
            f"  Posts:           [cyan]{stats['posts']}[/]\n"
            f"  Relationships:   [cyan]{stats['relationships']}[/]\n"
            f"  Preferences:     [cyan]{stats['preferences']}[/]\n"
            f"  Total items:     [bold cyan]{stats['total_items']}[/]\n"
            f"  Last updated:    [dim]{last_updated}[/]"
        )

        print_panel(
            f"[bold {COLORS['orange_5']}]OpenClawMini — Continuing Training[/]\n\n"
            f"Found existing memory for: [bold cyan]{config.user.name or memory.user.name or 'Unknown'}[/]\n\n"
            + stats_lines,
            title="🟠 OpenClawMini",
        )

        console.print(f"\n[bold {COLORS['orange_2']}]What would you like to do?[/]")
        console.print(f"  [{COLORS['orange_4']}][1][/] Continue training with existing memory")
        console.print(f"  [{COLORS['orange_4']}][2][/] Refresh memory (research again + merge)")
        console.print(f"  [{COLORS['orange_4']}][3][/] Upload new files to add to memory")
        console.print(f"  [{COLORS['orange_4']}][4][/] Start fresh (delete memory and restart)\n")

        choice = _prompt_int("  Your choice", 1)
        _handle_run_choice(choice, config)
    else:
        # First run — no memory.json yet
        from openclawmini.memory import MemoryStore, Memory
        store = MemoryStore(config.memory_file_path)
        memory = Memory()
        memory.user.name = config.user.name
        memory.user.email = config.user.email

        print_panel(
            f"[bold {COLORS['orange_5']}]OpenClawMini — First Run[/]\n\n"
            f"No memory found. Starting fresh research pipeline.\n"
            f"Configured sources: {', '.join(k for k, v in (config.user.data_sources or {}).items() if v) or 'none'}",
            title="🟠 OpenClawMini",
        )
        _run_research(config, store, memory, merge=False)
        _start_pipeline(config, store, memory)


def _handle_run_choice(choice: int, config: Config) -> None:
    memory_path = Path(config.memory_file_path)
    from openclawmini.memory import MemoryStore, Memory
    store = MemoryStore(str(memory_path))

    if choice == 1:
        console.print(f"\n[{COLORS['orange_3']}]Continuing with existing memory...[/]")
        memory = store.load()
        _start_pipeline(config, store, memory)
    elif choice == 2:
        console.print(f"\n[{COLORS['orange_3']}]Refreshing memory (research + merge)...[/]")
        memory = store.load()
        _run_research(config, store, memory, merge=True)
        _start_pipeline(config, store, memory)
    elif choice == 3:
        memory = store.load()
        _run_file_upload(config, store, memory)
    elif choice == 4:
        if Confirm.ask("  [red]Delete all memory and restart?[/]", default=False, console=console):
            memory_path.unlink(missing_ok=True)
            console.print(f"[{COLORS['orange_3']}]Memory cleared. Starting fresh research.[/]")
            memory = Memory()
            _run_research(config, store, memory, merge=False)
            _start_pipeline(config, store, memory)
        else:
            console.print("[dim]Cancelled.[/]")
    else:
        console.print("[red]Invalid choice.[/]")
        raise typer.Exit(1)


def _run_file_upload(config: Config, store: "MemoryStore", memory: "Memory") -> None:
    """Prompt user for file paths and process uploaded conversation logs."""
    from openclawmini.agents.research import ResearchAgent

    console.print(f"\n[bold {COLORS['orange_2']}]📂 File Upload[/]")
    console.print(
        f"[{COLORS['orange_4']}]Supported formats:[/] "
        f"ChatGPT export (conversations.json), "
        f"Claude export (claude_conversations.json), "
        f"generic role/content JSON lists.\n"
    )
    console.print(f"[dim]You can also upload any plain-text file (e.g. a resume, bio, or notes).[/]\n")

    file_paths: list[str] = []
    while True:
        path = Prompt.ask(
            f"  [bold]File path[/] [dim](leave blank to finish)[/]",
            default="",
            console=console,
        )
        if not path.strip():
            break
        p = Path(path.strip()).expanduser()
        if not p.exists():
            console.print(f"  [red]File not found: {p}[/]")
            continue
        file_paths.append(str(p))
        console.print(f"  [dim]Added: {p.name}[/]")

    if not file_paths:
        console.print(f"[{COLORS['orange_4']}]No files provided. Returning to pipeline.[/]")
        _start_pipeline(config, store, memory)
        return

    extractor = _build_gemini_extractor()
    if extractor:
        console.print(f"\n[dim]Gemini extractor active — extracting facts from conversations.[/]")
    else:
        console.print(f"\n[dim]No GOOGLE_API_KEY — using rule-based fact extraction.[/]")

    agent = ResearchAgent(store=store, memory=memory, llm_client=extractor)

    console.print(f"\n[{COLORS['orange_3']}]Processing {len(file_paths)} file(s)...[/]")
    result = agent.process_uploaded_files(file_paths)

    store.save(memory)

    if result.total_added > 0:
        console.print(f"\n[bold {COLORS['orange_5']}]✓ Files processed![/]")
        console.print(result.summary())
    else:
        console.print(f"\n[{COLORS['orange_4']}]No new items extracted from uploaded files.[/]")
    if result.errors:
        for err in result.errors:
            console.print(f"  [dim red]• {err}[/]")

    _start_pipeline(config, store, memory)


def _build_gemini_extractor():
    """Build a GeminiExtractor from env if GOOGLE_API_KEY is available."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openclawmini.integrations.gemini_extractor import GeminiExtractor
        return GeminiExtractor(api_key=api_key)
    except ImportError:
        return None


def _run_research(config: Config, store: "MemoryStore", memory: "Memory", merge: bool = False) -> None:
    """Run the Research Agent with live Rich progress output."""
    from openclawmini.agents.research import ResearchAgent
    from openclawmini.memory import Memory as Mem

    sources = config.user.data_sources or {}

    active_sources = [k for k, v in sources.items() if v]
    if not active_sources:
        console.print(f"[{COLORS['orange_4']}]No data sources configured. "
                      f"Re-run [bold cyan]openclawmini init[/] to set up sources.[/]")
        return

    console.print(f"\n[bold {COLORS['orange_2']}]🔍 Research Phase[/]")
    console.print(f"[dim]Sources: {', '.join(active_sources)}[/]\n")

    # Wire up Gemini for multi-fact extraction (requires GOOGLE_API_KEY)
    extractor = _build_gemini_extractor()
    if extractor:
        console.print(f"[dim]Gemini extractor active — will extract multiple facts per email.[/]\n")
    else:
        console.print(f"[dim]No GOOGLE_API_KEY found — using rule-based classification.[/]\n")

    agent = ResearchAgent(store=store, memory=memory, llm_client=extractor)

    # Live progress tracking via Rich
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    task_id = None
    progress = None
    current_stage = None

    _STAGE_LABEL = {
        "gmail": ("📧", "emails"),
        "gmail_extract": ("🧠", "batches"),
        "web_search": ("🔍", "queries"),
        "web_scrape": ("🌐", "pages"),
        "linkedin": ("💼", "profiles"),
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        nonlocal task_id, progress, current_stage
        if progress is None:
            return
        icon, unit = _STAGE_LABEL.get(stage, ("⏳", "items"))
        if stage != current_stage:
            # New stage — create a fresh task with the correct total
            current_stage = stage
            task_id = progress.add_task("", total=total)
        progress.update(
            task_id,
            completed=current,
            description=f"[bold {COLORS['orange_3']}]{icon} {stage}: {current}/{total} {unit}[/]",
        )

    with Progress(
        SpinnerColumn(style=COLORS["orange_2"]),
        TextColumn("[bold {task.description}]"),
        BarColumn(bar_width=40, style=COLORS["orange_4"], complete_style=COLORS["orange_2"]),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as prog:
        progress = prog
        result = agent.run(sources=sources, progress_callback=on_progress)

    # Save updated memory
    store.save(memory)

    # ── Document upload (BEFORE summary so docs enrich the total count) ───
    console.print(
        f"\n[bold {COLORS['orange_2']}]📄 Documents[/]\n"
        f"[dim]Share a resume, LinkedIn PDF, bio, or any text file to give the model "
        f"richer personal context. Gemini will extract 50+ facts from your resume.[/]\n"
    )
    if Confirm.ask("  Upload documents now?", default=True, console=console):
        _collect_document_files(store, memory, extractor)

    # Print result summary (after docs so totals include document facts)
    if result.total_added > 0 or len(getattr(memory, "facts", [])) > 0:
        console.print(f"\n[bold {COLORS['orange_5']}]✓ Research complete![/]")
        console.print(result.summary())
    else:
        console.print(f"\n[{COLORS['orange_4']}]No new items added to memory.[/]")
        if result.errors:
            for err in result.errors:
                console.print(f"  [dim red]• {err}[/]")

    if result.errors:
        console.print(f"\n[dim]Notes:[/]")
        for err in result.errors:
            console.print(f"  [dim]• {err}[/]")


def _collect_document_files(store, memory, extractor) -> None:
    """Prompt for file paths and process uploaded documents into memory."""
    from openclawmini.agents.research import ResearchAgent

    console.print(
        f"[dim]Enter file paths one at a time (tab-complete works). "
        f"Press Enter with no input when done.[/]\n"
    )
    file_paths: list[str] = []
    while True:
        raw = Prompt.ask(
            f"  [bold]File path[/] [dim](blank to finish)[/]",
            default="",
            console=console,
        )
        if not raw.strip():
            break
        # Strip shell escaping (e.g. drag-and-drop adds "\ " and "\(" on macOS)
        import re as _re
        cleaned = _re.sub(r"\\(.)", r"\1", raw.strip())
        p = Path(cleaned).expanduser()
        if not p.exists():
            console.print(f"  [red]File not found: {p}[/]")
            continue
        file_paths.append(str(p))
        console.print(f"  [dim]Added: {p.name}[/]")

    if not file_paths:
        console.print(f"[{COLORS['orange_4']}]No documents added.[/]\n")
        return

    agent = ResearchAgent(store=store, memory=memory, llm_client=extractor)
    console.print(f"\n[{COLORS['orange_3']}]Processing {len(file_paths)} document(s) with Gemini Flash...[/]")
    console.print(f"[dim]Extracting 50+ facts per document — this takes 30-60 seconds per file.[/]\n")
    doc_result = agent.process_uploaded_files(file_paths)

    # Also run extract_facts_from_document directly for richer resume extraction
    if extractor and hasattr(extractor, "extract_facts_from_document"):
        from openclawmini.memory.schema import DataSource
        from openclawmini.memory import MemoryStore
        for file_path in file_paths:
            try:
                p = Path(file_path)
                if p.suffix.lower() == ".pdf":
                    import pypdf
                    with open(file_path, "rb") as f:
                        reader = pypdf.PdfReader(f)
                        text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                else:
                    text = p.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    console.print(f"  [dim]Gemini extracting facts from {p.name}...[/]")
                    facts = extractor.extract_facts_from_document(
                        text=text,
                        filename=p.name,
                        user_name=getattr(memory.user, "name", ""),
                    )
                    added = 0
                    for fact in facts:
                        before = len(memory.facts)
                        store.add_fact(memory, fact)
                        if len(memory.facts) > before:
                            added += 1
                    if added:
                        console.print(f"  [bold {COLORS['orange_5']}]✓ {added} facts extracted from {p.name}[/]")
            except Exception as e:
                console.print(f"  [dim red]Document extraction error for {file_path}: {e}[/]")

    store.save(memory)

    if doc_result.total_added > 0 or len(getattr(memory, "facts", [])) > 0:
        console.print(f"\n[bold {COLORS['orange_5']}]✓ Documents processed![/]")
        if doc_result.total_added > 0:
            console.print(doc_result.summary())
    else:
        console.print(f"[{COLORS['orange_4']}]No new items extracted from documents.[/]")
    for err in doc_result.errors:
        console.print(f"  [dim red]• {err}[/]")


def _start_pipeline(config: Config, store=None, memory=None) -> None:
    """Pipeline entry point — generates eval set, optionally runs base eval."""
    console.print()
    stats = memory.stats() if memory else {}
    total = stats.get("total_items", 0)

    if total == 0:
        print_panel(
            f"[bold {COLORS['orange_5']}]Memory is empty![/]\n\n"
            f"Run research first (option 2) to collect your data,\n"
            f"or run [bold cyan]openclawmini init[/] to configure data sources.",
            title="🟠 Pipeline",
        )
        return

    # ── Generate / load eval set ───────────────────────────────
    _generate_eval_set(config, memory)


def _generate_eval_set(config: Config, memory) -> None:
    """Generate the persistent eval set from memory (or load existing)."""
    from openclawmini.agents.evals import EvalsAgent
    from openclawmini.eval.eval_set import EvalSetStore

    eval_store = EvalSetStore()
    extractor = _build_gemini_extractor()
    agent = EvalsAgent(gemini_extractor=extractor)

    if eval_store.exists():
        eval_set = eval_store.load()
        stats = eval_set.stats() if eval_set else {}
        console.print(
            f"[{COLORS['orange_3']}]📊 Eval set loaded:[/]  "
            f"[cyan]{stats.get('factual_questions', 0)}[/] factual questions, "
            f"[cyan]{stats.get('stylistic_prompts', 0)}[/] stylistic prompts\n"
        )
        want_regen = Confirm.ask(
            "  Regenerate eval set from updated memory?",
            default=False,
            console=console,
        )
        if not want_regen:
            _show_pipeline_status(config, memory, eval_set, stats)
            return
        eval_set = agent.generate_eval_set(memory, force=True)
    else:
        console.print(f"\n[bold {COLORS['orange_2']}]📊 Generating Eval Set[/]")
        console.print(f"[dim]Building factual questions + stylistic prompts from memory...[/]\n")
        eval_set = agent.generate_eval_set(memory)

    stats = eval_set.stats()
    console.print(f"[bold {COLORS['orange_5']}]✓ Eval set ready![/]")
    console.print(
        f"  Factual questions:  [cyan]{stats['factual_questions']}[/]  "
        f"(from {memory.stats()['facts']} facts)\n"
        f"  Stylistic prompts:  [cyan]{stats['stylistic_prompts']}[/]  "
        f"(from {memory.stats()['writing_samples']} writing samples)\n"
        f"  Saved → [dim]data/evals/eval_set.json[/]\n"
    )

    _show_pipeline_status(config, memory, eval_set, stats)


def _show_pipeline_status(config: Config, memory, eval_set, eval_stats: dict) -> None:
    """Show pipeline readiness and offer base model eval."""
    from openclawmini.agents.evals import EvalsAgent
    from openclawmini.eval.router import decide_next_action

    mem_stats = memory.stats()
    total = mem_stats.get("total_items", 0)

    print_panel(
        f"[bold {COLORS['orange_5']}]Pipeline ready[/]\n\n"
        f"  Memory:            [cyan]{total} items[/] collected\n"
        f"  Facts:             [cyan]{mem_stats['facts']}[/]\n"
        f"  Writing samples:   [cyan]{mem_stats['writing_samples']}[/]\n"
        f"  Eval questions:    [cyan]{eval_stats.get('factual_questions', 0)}[/] factual  "
        f"[cyan]{eval_stats.get('stylistic_prompts', 0)}[/] stylistic\n"
        f"  Orchestrator:      [cyan]{config.orchestrator.model}[/]\n"
        f"  Base model:        [cyan]{config.base_model.model}[/]\n"
        f"  SFT samples:       [cyan]{config.training.sft_sample_count}[/]\n"
        f"  GRPO targets:      [cyan]{config.training.grpo_scenario_count}[/]",
        title="🟠 Pipeline",
    )

    # Offer base model eval
    has_mistral_key = bool(os.getenv("MISTRAL_API_KEY", "").strip())
    if not has_mistral_key:
        console.print(
            f"[dim]Tip: Add MISTRAL_API_KEY to .env to run the base model eval.[/]\n"
        )
        _offer_data_generation(config, memory)
        return

    want_base_eval = Confirm.ask(
        "  Run base model eval now? (calls Mistral API)",
        default=False,
        console=console,
    )
    if want_base_eval:
        _run_base_eval(config)
    else:
        console.print(f"[dim]Skipping base eval.[/]\n")

    _offer_data_generation(config, memory)


def _offer_data_generation(config: Config, memory) -> None:
    """Check for existing training data and offer to generate (or regenerate) it."""
    training_dir = Path("./data/training")
    existing_sft = sorted(training_dir.glob("sft_*.jsonl")) if training_dir.exists() else []
    existing_grpo = sorted(training_dir.glob("grpo_*.jsonl")) if training_dir.exists() else []

    if existing_sft or existing_grpo:
        console.print(f"\n[{COLORS['orange_3']}]Training data found:[/]")
        if existing_sft:
            console.print(f"  SFT:  [dim]{existing_sft[-1].name}[/]")
        if existing_grpo:
            console.print(f"  GRPO: [dim]{existing_grpo[-1].name}[/]")
        want_regen = Confirm.ask(
            "  Regenerate training data from updated memory?",
            default=False,
            console=console,
        )
        if not want_regen:
            # Use existing data — still offer to run SFT if not yet trained
            if existing_sft:
                import types
                stub = types.SimpleNamespace(sft_path=str(existing_sft[-1]))
                _offer_sft_training(config, stub, memory)
            else:
                console.print(f"[dim]Using existing GRPO data. Generate SFT data first to run training.[/]\n")
            return
    else:
        want_data = Confirm.ask(
            "  Generate SFT + GRPO training data now?",
            default=True,
            console=console,
        )
        if not want_data:
            console.print(f"[dim]Skipping data generation. Run [bold cyan]openclawmini run[/] again to generate.[/]\n")
            return

    _run_data_cleansing(config, memory)


def _run_data_cleansing(config: Config, memory) -> None:
    """Generate SFT Q&A pairs and GRPO style scenarios from memory."""
    from openclawmini.agents.data_cleansing import DataCleansingAgent
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    mem_stats = memory.stats()

    console.print(f"\n[bold {COLORS['orange_2']}]📦 Training Data Generation[/]")
    console.print(
        f"[dim]Generating [cyan]{config.training.sft_sample_count}[/] SFT samples "
        f"from {mem_stats['facts']} facts, "
        f"and [cyan]{config.training.grpo_scenario_count}[/] GRPO scenarios "
        f"from {mem_stats['writing_samples']} writing samples.[/]\n"
    )

    extractor = _build_gemini_extractor()
    if extractor:
        console.print(f"[dim]Gemini active — generating high-quality training pairs.[/]\n")
    else:
        console.print(f"[dim]No GOOGLE_API_KEY — using template-based generation.[/]\n")

    user_name = config.user.name or memory.user.name or "the user"
    agent = DataCleansingAgent(
        llm_client=extractor,
        sft_target=config.training.sft_sample_count,
        grpo_target=config.training.grpo_scenario_count,
        quality_threshold=config.training.quality_threshold * 0.7,
        user_name=user_name,
    )

    sft_tid = None
    grpo_tid = None
    progress_obj = None

    def sft_progress(current: int, total: int) -> None:
        nonlocal sft_tid, progress_obj
        if progress_obj is None:
            return
        if sft_tid is None:
            sft_tid = progress_obj.add_task(
                f"[bold {COLORS['orange_3']}]SFT pairs...[/]", total=total
            )
        progress_obj.update(
            sft_tid, completed=current,
            description=f"[bold {COLORS['orange_3']}]SFT: {current}/{total} facts[/]",
        )

    def grpo_progress(current: int, total: int) -> None:
        nonlocal grpo_tid, progress_obj
        if progress_obj is None:
            return
        if grpo_tid is None:
            grpo_tid = progress_obj.add_task(
                f"[bold {COLORS['orange_3']}]GRPO scenarios...[/]", total=total
            )
        progress_obj.update(
            grpo_tid, completed=current,
            description=f"[bold {COLORS['orange_3']}]GRPO: {current}/{total} samples[/]",
        )

    try:
        with Progress(
            SpinnerColumn(style=COLORS["orange_2"]),
            TextColumn("[bold {task.description}]"),
            BarColumn(bar_width=40, style=COLORS["orange_4"], complete_style=COLORS["orange_2"]),
            MofNCompleteColumn(),
            console=console,
            transient=False,
        ) as prog:
            progress_obj = prog
            result = agent.run(
                memory=memory,
                sft_progress_callback=sft_progress,
                grpo_progress_callback=grpo_progress,
            )

        sft_pct = min(len(result.sft_samples) / max(config.training.sft_sample_count, 1), 1.0)
        grpo_pct = min(len(result.grpo_scenarios) / max(config.training.grpo_scenario_count, 1), 1.0)

        body = (
            f"[bold {COLORS['orange_5']}]✓ TRAINING DATA READY[/]\n\n"
            f"  SFT samples:    [cyan]{len(result.sft_samples)}[/]  {_ascii_bar(sft_pct)}\n"
            f"  GRPO scenarios: [cyan]{len(result.grpo_scenarios)}[/]  {_ascii_bar(grpo_pct)}\n\n"
            f"  From facts:     [dim]{result.facts_used}[/]\n"
            f"  From samples:   [dim]{result.samples_used}[/]\n"
            f"  From posts:     [dim]{result.posts_used}[/]"
        )
        if result.sft_path:
            body += f"\n\n  SFT  → [dim]{result.sft_path}[/]"
        if result.grpo_path:
            body += f"\n  GRPO → [dim]{result.grpo_path}[/]"

        # Log to W&B if configured
        try:
            from openclawmini.integrations.wb_logger import WBLogger
            WBLogger.from_env().log_training_data(result)
        except Exception:
            pass

        ds_tag = " [dim](via DataSimulator)[/]" if result.used_datasimulator else " [dim](template)[/]"
        body += f"\n  SFT method:{ds_tag}"
        print_panel(body, title="🟠 Data Cleansing")

        # Offer SFT training now that data is ready
        _offer_sft_training(config, result, memory)

    except Exception as e:
        console.print(f"[red]Data generation failed: {e}[/]")
        console.print(f"[dim]Check memory data and GOOGLE_API_KEY.[/]")


def _offer_sft_training(config: Config, data_result, memory) -> None:
    """After data gen, offer to run SFT training via ART."""
    from rich.prompt import Confirm

    has_wandb = bool(os.getenv("WANDB_API_KEY", "").strip())
    if not has_wandb:
        console.print(f"\n[dim]Tip: Add WANDB_API_KEY to .env to run SFT via ART serverless.[/]")
        return

    sft_path = data_result.sft_path
    if not sft_path or not Path(sft_path).exists():
        console.print(f"[dim]No SFT data file found — skipping training offer.[/]")
        return

    want_sft = Confirm.ask(
        "  Run SFT training now via ART serverless (CoreWeave GPU)?",
        default=True,
        console=console,
    )
    if not want_sft:
        console.print(f"[dim]Skipping SFT. Run again to train.[/]\n")
        return

    _run_sft_training(config, Path(sft_path), memory)


def _run_sft_training(config: Config, sft_path: Path, memory) -> None:
    """Run SFT training via ART, then eval, then offer GRPO."""
    from openclawmini.agents.sft_agent import SFTAgent
    from openclawmini.agents.evals import EvalsAgent

    user_name = config.user.name or memory.user.name or "the user"

    console.print(f"\n[bold {COLORS['orange_2']}]🎯 SFT Training — ART Serverless[/]")
    console.print(f"[dim]Model: {config.base_model.model}  Data: {sft_path.name}[/]\n")
    console.print(f"[dim]Compute runs on CoreWeave GPU via W&B ART. This may take several minutes.[/]\n")

    agent = SFTAgent.from_config(config, user_name=user_name)

    try:
        with console.status(f"[bold {COLORS['orange_3']}]SFT training in progress...[/]"):
            sft_result = agent.train(sft_path)

        print_panel(
            f"[bold {COLORS['orange_5']}]✓ SFT COMPLETE[/]\n\n"
            f"  Model:    [cyan]{sft_result.model_name}[/]\n"
            f"  Samples:  [cyan]{sft_result.samples_trained}[/]\n"
            f"  Project:  [dim]{sft_result.project}[/]",
            title="🟠 SFT",
        )

        # Post-SFT eval
        console.print(f"\n[bold {COLORS['orange_2']}]📊 Post-SFT Eval[/]")
        _run_stage_eval(config, stage="sft", model_name=sft_result.model_name)

        # Routing decision → offer GRPO
        _offer_grpo_training(config, sft_result, memory, user_name)

    except Exception as e:
        console.print(f"[red]SFT training failed: {e}[/]")
        console.print(f"[dim]Check WANDB_API_KEY and openpipe-art installation.[/]")


def _run_stage_eval(config: Config, stage: str, model_name: str):
    """Run factual + stylistic eval against an ART-trained model."""
    from openclawmini.agents.evals import EvalsAgent
    from openclawmini.eval.router import decide_next_action

    agent = EvalsAgent.from_env()

    # Build a model_fn that queries the ART model via its openai_client
    def art_model_fn(prompt: str) -> str:
        import asyncio
        import art  # type: ignore[import]

        model = art.TrainableModel(
            name=model_name,
            project=os.getenv("WANDB_PROJECT", "openclawmini"),
            base_model=config.base_model.model,   # mistral-small-2506
        )

        async def _query():
            backend = art.ServerlessBackend(api_key=os.getenv("WANDB_API_KEY", ""))
            await model.register(backend)
            client = model.openai_client()
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                max_tokens=256,
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""

        try:
            return asyncio.run(_query())
        except Exception:
            return ""

    try:
        results = agent.run_eval(model_fn=art_model_fn, stage=stage)

        action = decide_next_action(results, config)
        results.recommended_action = action.type

        factual_bar = _ascii_bar(results.factual_accuracy)
        stylistic_bar = _ascii_bar(results.stylistic_accuracy)

        print_panel(
            f"[bold {COLORS['orange_5']}]{stage.upper()} EVAL[/]\n\n"
            f"  Factual:    {results.factual_accuracy:.0%}  {factual_bar}\n"
            f"  Stylistic:  {results.stylistic_accuracy:.0%}  {stylistic_bar}\n"
            f"  Overall:    {results.overall_accuracy:.0%}  {_ascii_bar(results.overall_accuracy)}\n\n"
            f"→ Routing: [bold cyan]{action.type.upper()}[/]  {action.reason}",
            title=f"🟠 {stage.upper()} Eval",
        )

        try:
            from openclawmini.integrations.wb_logger import WBLogger
            WBLogger.from_env().log_eval(results)
        except Exception:
            pass

        return results

    except Exception as e:
        console.print(f"[red]Eval failed: {e}[/]")
        return None


def _offer_hf_upload(config: Config, project: str, model_name: str, user_name: str, stage: str, eval_result=None) -> None:
    """Ask the user if they want to upload the trained model to HuggingFace Hub."""
    from rich.prompt import Confirm
    from openclawmini.training.hf_exporter import ModelExporter

    exporter = ModelExporter.from_env() if hasattr(ModelExporter, "from_env") else ModelExporter()
    if not exporter.hf_token:
        console.print(f"[dim]HF_TOKEN not set — skipping HuggingFace upload offer.[/]\n")
        return

    metrics_str = ""
    if eval_result is not None:
        metrics_str = (
            f"\n  Factual: {eval_result.factual_accuracy:.0%}  "
            f"Stylistic: {eval_result.stylistic_accuracy:.0%}  "
            f"Overall: {eval_result.overall_accuracy:.0%}"
        )

    console.print(
        f"\n[bold {COLORS['orange_2']}]🤗 Upload to HuggingFace?[/]{metrics_str}"
    )
    want_upload = Confirm.ask(
        f"  Upload {stage.upper()} model to HuggingFace Hub?",
        default=True,
        console=console,
    )
    if not want_upload:
        console.print(f"[dim]Skipping HF upload.[/]\n")
        return

    _do_hf_upload(exporter, project, model_name, user_name, stage)


def _do_hf_upload(exporter, project: str, model_name: str, user_name: str, stage: str) -> None:
    """Run the HF upload and print the resulting repo URL."""
    console.print(f"[dim]Uploading {stage.upper()} adapter to HuggingFace...[/]")
    try:
        with console.status(f"[bold {COLORS['orange_3']}]Uploading to HuggingFace Hub...[/]"):
            repo_id = exporter.export(
                project=project,
                model_name=model_name,
                user_name=user_name,
                stage=stage,
            )
        if repo_id:
            print_panel(
                f"[bold {COLORS['orange_5']}]✓ Uploaded to HuggingFace[/]\n\n"
                f"  Repo:  [cyan]https://huggingface.co/{repo_id}[/]\n"
                f"  Stage: [cyan]{stage.upper()}[/]",
                title="🤗 HuggingFace",
            )
        else:
            console.print(f"[yellow]HF upload returned no repo ID — check HF_TOKEN and W&B artifact availability.[/]")
    except Exception as e:
        console.print(f"[red]HF upload failed: {e}[/]")


def _offer_grpo_training(config: Config, sft_result, memory, user_name: str) -> None:
    """After SFT eval, offer GRPO training."""
    from rich.prompt import Confirm

    grpo_files = sorted(Path("./data/training").glob("grpo_prompts_*.jsonl")) if Path("./data/training").exists() else []
    if not grpo_files:
        console.print(f"[dim]No GRPO prompts file found — run data generation first.[/]")
        return

    want_grpo = Confirm.ask(
        "  Run GRPO style-alignment training now?",
        default=True,
        console=console,
    )
    if not want_grpo:
        console.print(f"[dim]Skipping GRPO. Run again to train.[/]\n")
        return

    _run_grpo_training(config, sft_result=sft_result, memory=memory, user_name=user_name)


def _run_grpo_training(config: Config, sft_result, memory, user_name: str) -> None:
    """Run GRPO training via ART + RULER."""
    from openclawmini.agents.grpo_agent import GRPOAgent

    grpo_files = sorted(Path("./data/training").glob("grpo_prompts_*.jsonl"))
    grpo_path = grpo_files[-1]

    console.print(f"\n[bold {COLORS['orange_2']}]🎨 GRPO Training — ART + RULER[/]")
    console.print(f"[dim]Style alignment via online RL. Prompts: {grpo_path.name}[/]")
    console.print(f"[dim]RULER (Gemini Flash) scores each completion against {user_name}'s writing style.[/]\n")

    agent = GRPOAgent.from_sft_result(sft_result, config=config, user_name=user_name)

    try:
        with console.status(f"[bold {COLORS['orange_3']}]GRPO training in progress...[/]"):
            grpo_result = agent.train(grpo_path, memory=memory, user_name=user_name)

        print_panel(
            f"[bold {COLORS['orange_5']}]✓ GRPO COMPLETE[/]\n\n"
            f"  Model:    [cyan]{grpo_result.model_name}[/]\n"
            f"  Prompts:  [cyan]{grpo_result.prompts_trained}[/]\n"
            f"  Steps:    [cyan]{grpo_result.total_steps}[/]\n"
            f"  Avg RULER reward: [cyan]{grpo_result.final_reward:.3f}[/]",
            title="🟠 GRPO",
        )

        # Post-GRPO eval
        console.print(f"\n[bold {COLORS['orange_2']}]📊 Post-GRPO Eval[/]")
        eval_result = _run_stage_eval(config, stage="grpo", model_name=grpo_result.model_name)

        # Offer HuggingFace upload
        _offer_hf_upload(
            config=config,
            project=grpo_result.project,
            model_name=grpo_result.model_name,
            user_name=user_name,
            stage="grpo",
            eval_result=eval_result,
        )

    except Exception as e:
        console.print(f"[red]GRPO training failed: {e}[/]")
        console.print(f"[dim]Check WANDB_API_KEY, GOOGLE_API_KEY, and openpipe-art installation.[/]")


def _run_base_eval(config: Config) -> None:
    """Run base model eval and display results with routing recommendation."""
    from openclawmini.agents.evals import EvalsAgent
    from openclawmini.eval.router import decide_next_action
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    agent = EvalsAgent.from_env()

    console.print(f"\n[bold {COLORS['orange_2']}]🎯 Base Model Eval[/]")
    console.print(f"[dim]Model: {config.base_model.model}  (Mistral API)[/]\n")

    task_ids: dict = {}
    progress_obj = None

    def on_progress(dimension: str, current: int, total: int) -> None:
        nonlocal progress_obj
        if progress_obj is None:
            return
        if dimension not in task_ids:
            task_ids[dimension] = progress_obj.add_task(
                f"[bold {COLORS['orange_3']}]{dimension.title()} eval...[/]",
                total=total,
            )
        progress_obj.update(
            task_ids[dimension],
            completed=current,
            description=f"[bold {COLORS['orange_3']}]{dimension.title()}: {current}/{total}[/]",
        )

    try:
        with Progress(
            SpinnerColumn(style=COLORS["orange_2"]),
            TextColumn("[bold {task.description}]"),
            BarColumn(bar_width=40, style=COLORS["orange_4"], complete_style=COLORS["orange_2"]),
            MofNCompleteColumn(),
            console=console,
            transient=False,
        ) as prog:
            progress_obj = prog
            results = agent.run_base_eval(progress_callback=on_progress)

        action = agent.get_routing_action(results, config)
        results.recommended_action = action.type

        # Display results panel
        factual_bar = _ascii_bar(results.factual_accuracy)
        stylistic_bar = _ascii_bar(results.stylistic_accuracy)
        overall_bar = _ascii_bar(results.overall_accuracy)

        print_panel(
            f"[bold {COLORS['orange_5']}]BASE MODEL EVAL[/]\n\n"
            f"  Factual Accuracy:    {results.factual_accuracy:.0%}  {factual_bar}\n"
            f"  Stylistic Accuracy:  {results.stylistic_accuracy:.0%}  {stylistic_bar}\n"
            f"  Overall:             {results.overall_accuracy:.0%}  {overall_bar}\n\n"
            f"  ({results.factual_correct}/{results.factual_total} factual correct, "
            f"{results.stylistic_total} stylistic scored)\n\n"
            f"→ Routing decision: [bold cyan]{action.type.upper()}[/]\n"
            f"  {action.reason}",
            title="🟠 Eval Results",
        )

        console.print(
            f"[dim]Results saved → data/evals/results/base_*.json[/]\n"
            f"[dim]Data generation → SFT → GRPO coming in Tasks 6-8.[/]\n"
        )

    except Exception as e:
        console.print(f"[red]Base eval failed: {e}[/]")
        console.print(f"[dim]Check MISTRAL_API_KEY and eval set in data/evals/eval_set.json[/]")


def _ascii_bar(value: float, width: int = 12) -> str:
    """Return a simple ASCII progress bar for a 0-1 value."""
    filled = round(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


@app.command("train")
def cmd_train(
    budget: float = typer.Option(None, "--budget", "-b", help="USD budget cap (overrides config)."),
) -> None:
    """Collect memory, then run the autonomous SFT→GRPO loop until target accuracy is reached.

    Phase 1 (interactive): Gmail OAuth, web research, file uploads → builds memory.json
    Phase 2 (autonomous):  OrchestratorAgent dynamically decides SFT vs GRPO each round
    """
    load_env()
    print_banner()

    config = load_config()
    if config is None:
        console.print(
            f"[red]No config found.[/] Run [bold cyan]openclawmini init[/] first."
        )
        raise typer.Exit(1)

    if budget is not None:
        config.training.orchestrator_budget = budget

    from openclawmini.memory import MemoryStore, Memory

    memory_path = Path(config.memory_file_path)
    store = MemoryStore(str(memory_path))

    # ── PHASE 1: Memory collection ─────────────────────────────
    print_panel(
        f"[bold {COLORS['orange_5']}]Phase 1 — Memory Collection[/]\n\n"
        f"[dim]We'll gather your data from configured sources, then hand off\n"
        f"to the autonomous orchestrator for training.[/]",
        title="🟠 OpenClawMini Train",
    )

    if memory_path.exists():
        memory = store.load()
        stats = memory.stats()
        last_updated = memory.user.last_updated.strftime("%Y-%m-%d %H:%M") if memory.user.last_updated else "unknown"

        console.print(
            f"\n[{COLORS['orange_3']}]Existing memory found:[/]  "
            f"[cyan]{stats['total_items']} items[/]  "
            f"[dim](last updated {last_updated})[/]\n"
        )
        console.print(f"  [{COLORS['orange_4']}][1][/] Use existing memory and go straight to training")
        console.print(f"  [{COLORS['orange_4']}][2][/] Refresh memory first (re-research + merge), then train")
        console.print(f"  [{COLORS['orange_4']}][3][/] Start fresh (delete memory, research from scratch, then train)\n")

        choice = _prompt_int("  Your choice", 1)

        if choice == 2:
            _run_research(config, store, memory, merge=True)
        elif choice == 3:
            if Confirm.ask("  [red]Delete all memory and start fresh?[/]", default=False, console=console):
                memory_path.unlink(missing_ok=True)
                memory = Memory()
                memory.user.name = config.user.name
                memory.user.email = config.user.email
                _run_research(config, store, memory, merge=False)
            else:
                console.print("[dim]Cancelled.[/]")
                raise typer.Exit(0)
        # choice == 1: use existing, fall through
    else:
        # First run — no memory yet, must research
        memory = Memory()
        memory.user.name = config.user.name
        memory.user.email = config.user.email
        console.print(f"\n[{COLORS['orange_3']}]No memory found — starting research...[/]\n")
        _run_research(config, store, memory, merge=False)

    mem_stats = memory.stats()
    if mem_stats.get("total_items", 0) == 0:
        console.print(
            f"[red]Memory is still empty after research.[/] "
            f"Check your data sources in [dim]config.yaml[/] and API keys in [dim].env[/]."
        )
        raise typer.Exit(1)

    # ── Generate eval set silently (no interactive prompts) ────
    console.print(f"\n[bold {COLORS['orange_2']}]📊 Generating Eval Set[/]")
    console.print(f"[dim]Building factual questions + stylistic prompts from memory...[/]\n")
    _ensure_eval_set(config, memory)

    # ── PHASE 2: Autonomous training loop ─────────────────────
    print_panel(
        f"[bold {COLORS['orange_5']}]Phase 2 — Autonomous Training Loop[/]\n\n"
        f"  Memory:      [cyan]{mem_stats['total_items']} items[/] "
        f"({mem_stats['facts']} facts, {mem_stats['writing_samples']} writing samples)\n"
        f"  Target:      [cyan]{config.training.final_target_accuracy:.0%}[/] overall accuracy\n"
        f"  Budget:      [cyan]${config.training.orchestrator_budget:.0f}[/] USD\n"
        f"  Orchestrator:[cyan]{config.orchestrator.model}[/]\n"
        f"  Base model:  [cyan]{config.base_model.model}[/]\n\n"
        f"[dim]The orchestrator will now autonomously evaluate the base model,\n"
        f"generate training data, run SFT and GRPO, re-evaluate, and repeat\n"
        f"until the target accuracy is reached or the budget is exhausted.[/]",
        title="🟠 Autonomous Loop",
    )

    from openclawmini.agents.orchestrator_agent import OrchestratorAgent

    orchestrator = OrchestratorAgent.from_env(config, memory, store)

    try:
        with console.status(f"[bold {COLORS['orange_3']}]Orchestrator running autonomously...[/]"):
            result = orchestrator.run()
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Training interrupted.[/]  Progress saved to data/orchestrator/state.json")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Training loop failed: {e}[/]")
        raise typer.Exit(1)

    # ── Final results ──────────────────────────────────────────
    target_str = f"[bold green]Yes ✓[/]" if result.target_reached else f"[yellow]No[/]"
    print_panel(
        f"[bold {COLORS['orange_5']}]{'🎉 TARGET REACHED!' if result.target_reached else 'Training Complete'}[/]\n\n"
        + "\n".join(result.summary_lines())
        + f"\n  Target reached:     {target_str}",
        title="🟠 Results",
    )

    if result.eval_history:
        print_progress_table(result.eval_history)

    try:
        from openclawmini.integrations.wb_logger import WBLogger
        logger = WBLogger.from_env()
        for entry in result.eval_history:
            if hasattr(entry, "factual_accuracy"):
                logger.log_eval(entry)
    except Exception:
        pass

    # Auto-upload to HuggingFace when the model is approved (target reached)
    if result.target_reached and result.final_model_name:
        from openclawmini.training.hf_exporter import ModelExporter
        exporter = ModelExporter.from_env()
        if exporter.hf_token:
            console.print(f"\n[bold {COLORS['orange_2']}]🤗 Target reached — uploading model to HuggingFace...[/]")
            _do_hf_upload(
                exporter=exporter,
                project=result.final_project or "openclawmini",
                model_name=result.final_model_name,
                user_name=config.user.name,
                stage=result.best_stage or "grpo",
            )
        else:
            console.print(f"[dim]Set HF_TOKEN to auto-upload trained models to HuggingFace.[/]")


def _ensure_eval_set(config: Config, memory) -> None:
    """Generate eval set from memory, or load existing one — no interactive prompts."""
    from openclawmini.agents.evals import EvalsAgent
    from openclawmini.eval.eval_set import EvalSetStore

    eval_store = EvalSetStore()
    extractor = _build_gemini_extractor()
    agent = EvalsAgent(gemini_extractor=extractor)

    if eval_store.exists():
        eval_set = eval_store.load()
        stats = eval_set.stats() if eval_set else {}
        # Regenerate if we now have facts but the saved eval set has none —
        # this happens when the eval set was created before research completed.
        has_facts = len(getattr(memory, "facts", [])) > 0
        needs_regen = has_facts and stats.get("factual_questions", 0) == 0
        if needs_regen:
            console.print(
                f"[{COLORS['orange_4']}]Eval set has 0 factual questions but memory now has facts "
                f"— regenerating...[/]\n"
            )
            eval_store.delete()
            eval_set = agent.generate_eval_set(memory)
            stats = eval_set.stats()
            console.print(
                f"[bold {COLORS['orange_5']}]✓ Eval set ready:[/]  "
                f"[cyan]{stats.get('factual_questions', 0)}[/] factual,  "
                f"[cyan]{stats.get('stylistic_prompts', 0)}[/] stylistic\n"
            )
        else:
            console.print(
                f"[{COLORS['orange_3']}]Eval set loaded:[/]  "
                f"[cyan]{stats.get('factual_questions', 0)}[/] factual,  "
                f"[cyan]{stats.get('stylistic_prompts', 0)}[/] stylistic\n"
            )
    else:
        eval_set = agent.generate_eval_set(memory)
        stats = eval_set.stats()
        console.print(
            f"[bold {COLORS['orange_5']}]✓ Eval set ready:[/]  "
            f"[cyan]{stats.get('factual_questions', 0)}[/] factual,  "
            f"[cyan]{stats.get('stylistic_prompts', 0)}[/] stylistic\n"
        )


@app.command("chat")
def cmd_chat() -> None:
    """Chat with your trained personalized model."""
    load_env()
    print_banner()
    print_panel(
        f"[bold {COLORS['orange_5']}]Chat mode coming soon![/]\n\n"
        f"[dim]Run [bold cyan]openclawmini run[/] first to train your model.[/]",
        title="🟠 OpenClawMini Chat",
    )

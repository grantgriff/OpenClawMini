# OpenClawMini

**Your Personal AI That Learns To Be You.**

OpenClawMini fine-tunes a large language model on your personal data — emails, web presence, LinkedIn, documents — until it can answer factual questions about you and write in your voice. The entire pipeline is autonomous: a Claude-powered orchestrator decides when to research, when to train, and when to stop.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1 — Memory Collection                                        │
│                                                                     │
│  Gmail ──┐                                                          │
│  Web ────┤──► ResearchAgent ──► GeminiExtractor ──► memory.json    │
│  LinkedIn┤        │                                                 │
│  Files ──┘        └──► writing samples, facts, relationships       │
│                                                                     │
│  (Interactive — user approves sources, uploads docs, etc.)         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Eval Set Generation                                                │
│                                                                     │
│  memory.json ──► EvalsAgent ──► factual Q&A + stylistic prompts    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2 — Autonomous Training Loop                                 │
│                                                                     │
│  OrchestratorAgent (Claude Opus)                                    │
│    │                                                                │
│    ├── run_eval()       ──► EvalsAgent ──► factual + style scores  │
│    ├── targeted_research() ──► ResearchAgent (web, GitHub)         │
│    ├── generate_sft_data() ──► DataCleansing + DataSimulator       │
│    ├── run_sft()        ──► SFTAgent ──► ART/W&B CoreWeave GPU     │
│    ├── run_grpo()       ──► GRPOAgent ──► ART/W&B CoreWeave GPU    │
│    └── stop()           ──► target reached or budget exhausted     │
│                                                                     │
│  (Fully autonomous — loops until accuracy target or budget cap)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Three Commands

### `openclawmini init`
Interactive first-run setup. Walks you through connecting data sources (Gmail OAuth, LinkedIn, web search), choosing your AI models, and setting API keys. Writes `config.yaml` and `.env`.

### `openclawmini train`
The main command. Runs the full two-phase pipeline end-to-end:
- **Phase 1** collects your personal data into `memory.json`
- **Phase 2** runs the autonomous SFT → GRPO training loop until your target accuracy is reached or your budget is exhausted

```bash
openclawmini train
openclawmini train --budget 15   # cap spend at $15
```

### `openclawmini run`
Manual one-pass wizard. Runs research → SFT → GRPO as a single pipeline pass with interactive prompts at each step. Useful for debugging individual stages or doing a controlled single training run.

---

## Architecture

### Agents

#### `ResearchAgent`
Collects personal facts and writing samples from configured data sources. In Phase 1 it does a broad sweep; the orchestrator can also call it surgically on a specific weak category (e.g. just `education`) between training rounds.

Data sources:
- **Gmail** — OAuth2, reads sent mail, extracts facts via Gemini (batch extraction, multi-fact per email)
- **Web search** — DuckDuckGo queries on the user's name across 8 categories, parallel scraping
- **LinkedIn** — scrapes public profile page
- **GitHub** — fetches profile for skills/work/achievements enrichment
- **File uploads** — PDF, text, markdown (resumes, bios, LinkedIn exports)

Fact extraction uses `GeminiExtractor` (Gemini 2.5 Flash) which returns structured `Fact` objects with `category`, `confidence`, and `source`. Facts are stored nameless ("Works at Acme Corp", not "Grant works at Acme Corp") to avoid LLM safety filter issues downstream.

#### `EvalsAgent`
Builds and runs the evaluation suite. Generates two types of questions from `memory.json`:
- **Factual** — "Where did [person] go to university?" — scored by Gemini judge (0/1)
- **Stylistic** — open-ended writing prompts scored by a RULER-based style similarity judge

Tracks per-category accuracy breakdowns (work, education, skills, etc.) so the orchestrator knows exactly where the model is weak.

#### `OrchestratorAgent`
The brain of Phase 2. Powered by **Claude Opus** via the Anthropic API, it runs a tool-use loop with a $25 (configurable) budget. Each round it receives the eval history, training history, and research history, then decides which tool to call next:

| Tool | When used | Approx cost |
|---|---|---|
| `run_eval` | Always first; after each training round | ~$0.25 |
| `targeted_research` | Only when a category is still weak after SFT | ~$0.25 |
| `generate_sft_data` | When factual accuracy < 70% | ~$1.00 |
| `run_sft` | After data generation | free (W&B GPU) |
| `run_grpo` | When factual is ≥ threshold but style lags | free (W&B GPU) |
| `stop` | Target reached, or 3 rounds with no improvement | — |

Falls back to a deterministic rule-based routing if no `ANTHROPIC_API_KEY` is set.

#### `SFTAgent`
Converts `memory.json` facts into `{"messages": [...]}` JSONL training pairs via `DataCleansing` + DataSimulator, then submits them to **ART** (OpenPipe Agent Reinforcement Trainer) for supervised fine-tuning on W&B's serverless CoreWeave GPU cluster.

Base model: `OpenPipe/Qwen3-14B-Instruct` (HuggingFace). Training is free — only Gemini API calls for data generation cost money.

#### `GRPOAgent`
After SFT, runs GRPO (Group Relative Policy Optimization) reinforcement learning to improve stylistic quality. Uses **RULER** — a writing style similarity scorer — as the reward signal. Also runs on ART/CoreWeave serverless.

#### `DataCleansing` + `DataSimulator`
Transforms raw facts and writing samples into clean training data:
- `DataCleansing` filters, deduplicates, and formats facts into SFT-ready JSONL
- `DataSimulator` (external SDK) uses Gemini to generate diverse Q&A pairs, conversation scenarios, and stylistic prompts from the cleaned facts

---

### Memory Schema

`memory.json` holds everything learned about the user:

```
Memory
 ├── facts[]          — structured facts (category, confidence, source)
 ├── writing_samples[] — raw text samples for style learning
 ├── relationships[]   — people the user knows
 └── user             — name, email, last_updated
```

---

### Model Stack

| Role | Model | API |
|---|---|---|
| Base model (fine-tuned) | `OpenPipe/Qwen3-14B-Instruct` | HuggingFace Inference API |
| Fact extraction | Gemini 2.5 Flash | Google AI |
| Data generation | Gemini 2.5 Pro | Google AI |
| Eval judge | Gemini 2.5 Flash | Google AI |
| GRPO reward (RULER) | Gemini 2.5 Flash | Google AI |
| Orchestrator | Claude Opus 4.6 | Anthropic API |
| Training backend | W&B Serverless RL (CoreWeave) | Weights & Biases |

---

## Setup

### 1. Install

```bash
git clone <repo>
cd OpenClawMini
pip install -e .
```

### 2. Configure API keys

Create a `.env` file:

```env
# Required
GOOGLE_API_KEY=...        # Gemini (research, data gen, eval judge)
HF_TOKEN=...              # HuggingFace (base model inference + model upload)
WANDB_API_KEY=...         # W&B (ART fine-tuning backend)

# Optional — enables Claude orchestrator (falls back to rule-based without it)
ANTHROPIC_API_KEY=...

# Optional — Gmail research source
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
```

### 3. Run init

```bash
openclawmini init
```

This walks you through data source setup and writes `config.yaml`.

### 4. Train

```bash
openclawmini train
```

---

## Configuration

`config.yaml` controls all model choices and training hyperparameters:

```yaml
models:
  orchestrator:
    provider: "anthropic"
    model: "claude-opus-4-6"

  research_extraction:
    provider: "google"
    model: "gemini-2.5-flash"

  data_generation:
    provider: "google"
    model: "gemini-2.5-pro"

  eval_judge:
    provider: "google"
    model: "gemini-2.5-flash"

  # Base model to fine-tune
  base_model:
    provider: "openpipe"
    model: "OpenPipe/Qwen3-14B-Instruct"

training:
  sft_sample_count: 30          # Q&A pairs per SFT run
  grpo_scenario_count: 15       # Writing scenarios per GRPO run
  sft_factual_threshold: 0.70   # Factual accuracy before switching to GRPO
  final_target_accuracy: 0.80   # Stop when overall accuracy hits this
  orchestrator_budget: 25.0     # Max USD for the autonomous loop
```

---

## Cost Breakdown

| Operation | Cost |
|---|---|
| Phase 1 research (web + Gmail) | ~$0.50–1.00 |
| SFT data generation (DataSimulator) | ~$1.00 |
| SFT training (W&B CoreWeave) | **free** |
| GRPO training (W&B CoreWeave) | **free** |
| Eval round (Gemini judge) | ~$0.25 |
| Claude orchestrator per round | ~$0.10 |
| **Full BASE→SFT→GRPO loop** | **~$2–4** |
| **Typical full run (multiple loops)** | **~$8–15** |

---

## Project Structure

```
openclawmini/
├── agents/
│   ├── orchestrator_agent.py   # Claude-powered autonomous loop
│   ├── research.py             # Web/Gmail/LinkedIn data collection
│   ├── evals.py                # Factual + stylistic evaluation
│   ├── sft_agent.py            # Supervised fine-tuning via ART
│   ├── grpo_agent.py           # GRPO RL training via ART
│   └── data_cleansing.py       # Fact → training data pipeline
├── integrations/
│   ├── gemini_extractor.py     # Multi-fact Gemini extraction
│   └── mistral_extractor.py    # Legacy rule-based extractor
├── memory/
│   └── schema.py               # Fact, WritingSample, Memory dataclasses
├── training/
│   ├── sft_generator.py        # SFT JSONL generation
│   ├── grpo_generator.py       # GRPO scenario generation
│   └── ruler.py                # Writing style reward scorer
├── utils/
│   └── llm_client.py           # OpenPipeClient (HF inference), MistralClient
├── config.py                   # Config dataclasses + loader
└── cli.py                      # Typer CLI (init, train, run, chat)
```

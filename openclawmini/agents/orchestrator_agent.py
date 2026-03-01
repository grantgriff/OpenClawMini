"""
OrchestratorAgent — autonomous fine-tuning pipeline manager using Claude Opus.

Manages the full BASE → SFT → GRPO training loop autonomously via Claude tool use:
  1. Evaluate current model (factual + stylistic accuracy, per-category breakdown)
  2. Identify weak categories via per-category accuracy
  3. Run targeted research on weak categories if needed
  4. Generate or reuse SFT training data
  5. Run SFT or GRPO training based on eval results
  6. Re-evaluate and decide next step
  7. Stop when target accuracy reached or budget exhausted

State persistence:
  data/orchestrator/state.json — tracks eval history, training history, cost

Auth:
  ANTHROPIC_API_KEY — Claude Opus tool-use backbone
  Falls back to rule-based routing if key not set

Budget: configurable via config.training.orchestrator_budget (default $25)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel


# ── Result dataclasses ────────────────────────────────────────


@dataclass
class OrchestratorResult:
    rounds_completed: int
    final_overall_accuracy: float
    final_factual_accuracy: float
    final_stylistic_accuracy: float
    best_stage: str
    cost_usd: float
    target_reached: bool
    stop_reason: str
    eval_history: list = field(default_factory=list)
    final_model_name: str = ""
    final_project: str = ""

    def summary_lines(self) -> list[str]:
        return [
            f"  Rounds completed:   {self.rounds_completed}",
            f"  Final accuracy:     {self.final_overall_accuracy:.0%}",
            f"  Factual accuracy:   {self.final_factual_accuracy:.0%}",
            f"  Stylistic accuracy: {self.final_stylistic_accuracy:.0%}",
            f"  Cost:               ${self.cost_usd:.2f}",
            f"  Target reached:     {'Yes ✓' if self.target_reached else 'No'}",
            f"  Stop reason:        {self.stop_reason}",
        ]


# ── OrchestratorAgent ─────────────────────────────────────────


class OrchestratorAgent:
    """
    Autonomous fine-tuning orchestrator using Claude Opus tool use.

    Manages the full training pipeline autonomously by calling tools
    (run_eval, run_sft, run_grpo, targeted_research, generate_sft_data,
    check_status, stop) in a loop until target accuracy is reached
    or the budget is exhausted.

    Falls back to rule-based routing if ANTHROPIC_API_KEY is not set.

    Args:
        config:          Config object
        memory:          Memory object (user facts + writing samples)
        store:           MemoryStore for persisting memory updates
        evals_agent:     EvalsAgent for running evaluations
        data_agent:      DataCleansingAgent for generating training data
        sft_agent:       SFTAgent for supervised fine-tuning
        research_agent:  ResearchAgent for targeted research
        budget:          Max USD to spend (overrides config.training.orchestrator_budget)
        anthropic_api_key: Falls back to ANTHROPIC_API_KEY env var
    """

    SYSTEM_PROMPT = """\
You are an autonomous AI fine-tuning orchestrator. Your job is to improve \
a personalized AI model for {user_name} to reach {target_accuracy:.0%} overall accuracy \
within a ${budget:.0f} budget.

You control a fine-tuning pipeline with these tools:
- run_eval: Evaluate current model accuracy (factual + stylistic + per-category)
- run_sft: Run supervised fine-tuning (W&B compute FREE; cost ~$1.50 for data gen)
- run_grpo: Run GRPO RL training (W&B compute FREE; cost ~$0.50 for RULER scoring)
- targeted_research: Targeted web research on a weak category (~$0.25)
- generate_sft_data: Generate new SFT training data (~$1.00 for DataSimulator)
- check_status: Check budget, training history, current model
- stop: Stop when done

Cost reality: W&B/CoreWeave training is FREE. Costs come from Gemini API calls
(DataSimulator, RULER, eval judge) and Claude Opus (this orchestrator).
A full BASE→SFT→GRPO loop costs approximately $2-4. With a ${budget:.0f} budget
you can run 6+ complete loops.

Decision guidelines:
1. Always start with run_eval("base") if no eval history exists
2. If factual accuracy < {sft_threshold:.0%}: research weak categories → generate_sft_data → run_sft
3. If factual accuracy >= threshold but overall < target: run_grpo for stylistic improvement
4. Target specific weak categories with targeted_research before retraining
5. Stop when target reached OR after 3 consecutive rounds with no improvement
6. Be decisive — one action per round, no overthinking
"""

    def __init__(
        self,
        config,
        memory,
        store,
        evals_agent=None,
        data_agent=None,
        sft_agent=None,
        research_agent=None,
        budget: Optional[float] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.store = store
        self.evals_agent = evals_agent
        self.data_agent = data_agent
        self.sft_agent = sft_agent
        self.research_agent = research_agent

        self._budget = budget or getattr(config.training, "orchestrator_budget", 25.0)
        self._api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._console = Console()

        # Mutable training state
        self._current_sft_result = None
        self._current_grpo_result = None
        self._current_model_name = config.base_model.model
        self._current_stage = "base"
        self._stopped = False
        self._stop_reason = ""

        # Persistent state
        self._state_path = Path("./data/orchestrator/state.json")
        self._state = self._load_state()

    # ── Public API ──────────────────────────────────────────────

    def run(self) -> OrchestratorResult:
        """
        Run the autonomous fine-tuning loop.

        Blocks until target accuracy reached, budget exhausted, or Claude stops.
        """
        user_name = getattr(self.config.user, "name", "the user")
        target = getattr(self.config.training, "final_target_accuracy", 0.80)

        self._console.print(Panel(
            f"[bold]Autonomous Fine-Tuning Orchestrator[/bold]\n"
            f"User:    {user_name}\n"
            f"Target:  {target:.0%} overall accuracy\n"
            f"Budget:  ${self._budget:.0f}\n"
            f"Spent:   ${self._state.get('cost_usd', 0.0):.2f}",
            title="[bold yellow]Orchestrator Starting[/bold yellow]",
            border_style="yellow",
        ))

        if not self._api_key:
            self._console.print(
                "[yellow]ANTHROPIC_API_KEY not set — using rule-based routing[/yellow]"
            )
            return self._run_rule_based(user_name, target)

        return self._run_claude_loop(user_name, target)

    # ── Claude tool-use loop ────────────────────────────────────

    def _run_claude_loop(self, user_name: str, target: float) -> OrchestratorResult:
        """Main tool-use loop driven by Claude Opus."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for orchestrator. "
                "Run: pip install anthropic>=0.34"
            )

        client = anthropic.Anthropic(api_key=self._api_key)
        sft_threshold = getattr(self.config.training, "sft_factual_threshold", 0.70)

        system = self.SYSTEM_PROMPT.format(
            user_name=user_name,
            target_accuracy=target,
            budget=self._budget,
            sft_threshold=sft_threshold,
        )

        messages = [
            {"role": "user", "content": self._build_initial_message(user_name, target)}
        ]

        max_turns = 30  # Safety ceiling

        for _ in range(max_turns):
            if self._stopped:
                break

            spent = self._state.get("cost_usd", 0.0)
            if spent >= self._budget - 2.0:
                self._stop_reason = (
                    f"Budget nearly exhausted "
                    f"(${spent:.2f} / ${self._budget:.2f} spent)"
                )
                break

            try:
                response = client.messages.create(
                    model=self.config.orchestrator.model,
                    max_tokens=2048,
                    system=system,
                    messages=messages,
                    tools=self._build_tool_definitions(),
                )
            except Exception as e:
                self._stop_reason = f"Claude API error: {e}"
                break

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            # Claude finished without a tool call
            if response.stop_reason == "end_turn":
                self._stop_reason = self._stop_reason or "Claude decided training is complete"
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue

                result = self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

                if block.name == "stop":
                    self._stopped = True
                    self._stop_reason = block.input.get("reason", "Stopped by orchestrator")
                    break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        return self._build_final_result()

    # ── Tool execution ──────────────────────────────────────────

    def _execute_tool(self, name: str, inputs: dict) -> dict:
        """Dispatch tool call and return JSON-serializable result."""
        self._console.print(f"  [dim]→ {name}({', '.join(f'{k}={v!r}' for k, v in inputs.items())})[/dim]")

        try:
            if name == "run_eval":
                return self._tool_run_eval(inputs.get("stage", "current"))
            elif name == "run_sft":
                return self._tool_run_sft(inputs.get("sft_path", ""))
            elif name == "run_grpo":
                return self._tool_run_grpo()
            elif name == "targeted_research":
                return self._tool_targeted_research(
                    inputs.get("category", "other"),
                    inputs.get("additional_context", ""),
                )
            elif name == "generate_sft_data":
                return self._tool_generate_sft_data(
                    inputs.get("focus_categories", [])
                )
            elif name == "check_status":
                return self._tool_check_status()
            elif name == "stop":
                return {"stopped": True, "reason": inputs.get("reason", "")}
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e), "tool": name}

    def _tool_run_eval(self, stage: str) -> dict:
        if self.evals_agent is None:
            return {"error": "EvalsAgent not configured"}

        self._console.print(Panel(
            f"Evaluating {stage} model...",
            title=f"[bold cyan]Eval: {stage}[/bold cyan]",
            border_style="cyan",
        ))

        model_fn = self._build_model_fn()
        results = self.evals_agent.run_eval(model_fn=model_fn, stage=stage)

        result_dict = {
            "stage": stage,
            "factual_accuracy": results.factual_accuracy,
            "stylistic_accuracy": results.stylistic_accuracy,
            "overall_accuracy": results.overall_accuracy,
            "weak_dimension": results.weak_dimension,
            "weak_categories": getattr(results, "weak_categories", []),
            "category_accuracy": getattr(results, "category_accuracy", {}),
            "factual_correct": results.factual_correct,
            "factual_total": results.factual_total,
        }

        self._state.setdefault("eval_history", []).append(result_dict)
        self._save_state()

        self._console.print(Panel(
            "\n".join(results.summary_lines()),
            title=f"[bold green]Eval Results: {stage}[/bold green]",
            border_style="green",
        ))

        return result_dict

    def _tool_run_sft(self, sft_path: str = "") -> dict:
        if self.sft_agent is None:
            return {"error": "SFTAgent not configured"}

        if not sft_path:
            sft_files = sorted(Path("./data/training").glob("sft_*.jsonl"))
            if not sft_files:
                return {"error": "No SFT data found. Run generate_sft_data first."}
            sft_path = str(sft_files[-1])

        self._console.print(Panel(
            f"Training from: {Path(sft_path).name}",
            title="[bold magenta]SFT Training[/bold magenta]",
            border_style="magenta",
        ))

        sft_result = self.sft_agent.train(Path(sft_path))
        self._current_sft_result = sft_result
        self._current_model_name = sft_result.model_name
        self._current_stage = "sft"

        # W&B/CoreWeave SFT training is FREE (compute covered by W&B platform).
        # Real costs: Gemini DataSimulator calls (~$1-2) + Mistral eval (~$0.05)
        cost_est = 1.5
        self._state["cost_usd"] = self._state.get("cost_usd", 0.0) + cost_est
        self._state.setdefault("training_history", []).append({
            "action": "sft",
            "model_name": sft_result.model_name,
            "samples": sft_result.samples_trained,
            "cost_est": cost_est,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()

        self._console.print(Panel(
            f"Model: {sft_result.model_name}\nSamples: {sft_result.samples_trained}",
            title="[bold green]SFT Complete[/bold green]",
            border_style="green",
        ))

        return {
            "success": True,
            "model_name": sft_result.model_name,
            "samples_trained": sft_result.samples_trained,
            "estimated_cost_usd": cost_est,
        }

    def _tool_run_grpo(self) -> dict:
        if self._current_sft_result is None:
            return {"error": "No SFT model available. Run run_sft first."}

        from openclawmini.agents.grpo_agent import GRPOAgent

        user_name = getattr(self.config.user, "name", "")
        grpo_agent = GRPOAgent.from_sft_result(
            self._current_sft_result,
            config=self.config,
            user_name=user_name,
        )

        grpo_files = sorted(Path("./data/training").glob("grpo_prompts_*.jsonl"))
        if not grpo_files:
            return {"error": "No GRPO prompts found. Run generate_sft_data first."}

        self._console.print(Panel(
            f"Base: {self._current_sft_result.model_name}",
            title="[bold magenta]GRPO Training[/bold magenta]",
            border_style="magenta",
        ))

        grpo_result = grpo_agent.train(
            grpo_jsonl_path=grpo_files[-1],
            memory=self.memory,
            user_name=user_name,
        )
        self._current_grpo_result = grpo_result
        self._current_model_name = grpo_result.model_name
        self._current_stage = "grpo"

        # W&B/CoreWeave GRPO training is FREE (compute covered by W&B platform).
        # Real costs: Gemini Flash RULER scoring (~$0.20) + Gemini eval (~$0.10)
        cost_est = 0.50
        self._state["cost_usd"] = self._state.get("cost_usd", 0.0) + cost_est
        self._state.setdefault("training_history", []).append({
            "action": "grpo",
            "model_name": grpo_result.model_name,
            "final_reward": grpo_result.final_reward,
            "cost_est": cost_est,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()

        self._console.print(Panel(
            f"Model: {grpo_result.model_name}\nAvg reward: {grpo_result.final_reward:.3f}",
            title="[bold green]GRPO Complete[/bold green]",
            border_style="green",
        ))

        return {
            "success": True,
            "model_name": grpo_result.model_name,
            "final_reward": grpo_result.final_reward,
            "estimated_cost_usd": cost_est,
        }

    def _tool_targeted_research(
        self, category: str, additional_context: str = ""
    ) -> dict:
        if self.research_agent is None:
            return {"error": "ResearchAgent not configured"}

        user_name = getattr(self.config.user, "name", "")
        result = self.research_agent.targeted_research(
            user_name=user_name,
            category=category,
            memory=self.memory,
        )
        self.store.save(self.memory)

        cost_est = 0.50
        self._state["cost_usd"] = self._state.get("cost_usd", 0.0) + cost_est
        self._save_state()

        return {
            "facts_added": result.facts_added,
            "estimated_cost_usd": cost_est,
        }

    def _tool_generate_sft_data(self, focus_categories: list = None) -> dict:
        if self.data_agent is None:
            return {"error": "DataCleansingAgent not configured"}

        self._console.print("  [dim]Generating SFT + GRPO training data...[/dim]")
        cleansing_result = self.data_agent.run(self.memory)

        result = {
            "sft_samples": len(cleansing_result.sft_samples),
            "grpo_scenarios": len(cleansing_result.grpo_scenarios),
        }
        if cleansing_result.sft_path:
            result["sft_path"] = str(cleansing_result.sft_path)
        if cleansing_result.grpo_path:
            result["grpo_path"] = str(cleansing_result.grpo_path)
        return result

    def _tool_check_status(self) -> dict:
        spent = self._state.get("cost_usd", 0.0)
        return {
            "budget_total": self._budget,
            "budget_spent": round(spent, 2),
            "budget_remaining": round(self._budget - spent, 2),
            "current_model": self._current_model_name,
            "current_stage": self._current_stage,
            "training_rounds": len(self._state.get("training_history", [])),
            "eval_count": len(self._state.get("eval_history", [])),
            "last_eval": (
                self._state["eval_history"][-1]
                if self._state.get("eval_history") else None
            ),
        }

    # ── Rule-based fallback ─────────────────────────────────────

    def _run_rule_based(self, user_name: str, target: float) -> OrchestratorResult:
        """
        Rule-based routing fallback when ANTHROPIC_API_KEY is not set.
        Uses the existing decide_next_action() router from eval/router.py.
        """
        from openclawmini.eval.router import decide_next_action

        max_rounds = 4

        for round_num in range(max_rounds):
            spent = self._state.get("cost_usd", 0.0)
            if spent >= self._budget - 2.0:
                self._stop_reason = f"Budget exhausted (${spent:.2f} spent)"
                break

            # Eval
            eval_result = self._tool_run_eval(self._current_stage)
            overall = eval_result.get("overall_accuracy", 0.0)

            if overall >= target:
                self._stop_reason = f"Target accuracy reached: {overall:.0%}"
                break

            # Route using existing logic (recreate minimal duck-typed object)
            class _MockResults:
                pass

            mock = _MockResults()
            mock.overall_accuracy = overall
            mock.factual_accuracy = eval_result.get("factual_accuracy", 0.0)
            mock.stylistic_accuracy = eval_result.get("stylistic_accuracy", 0.0)
            mock.weak_dimension = eval_result.get("weak_dimension", "factual")

            action = decide_next_action(mock, self.config)

            if action.type == "complete":
                self._stop_reason = action.reason
                break
            elif action.type == "sft":
                # Targeted research on weak categories first
                for cat in eval_result.get("weak_categories", [])[:2]:
                    self._tool_targeted_research(cat)
                self._tool_generate_sft_data()
                self._tool_run_sft()
            elif action.type == "grpo":
                self._tool_run_grpo()

        self._stop_reason = self._stop_reason or "Max rounds reached"
        return self._build_final_result()

    # ── Helpers ─────────────────────────────────────────────────

    def _build_model_fn(self) -> Callable[[str], str]:
        """Build model inference function for the current training stage."""
        if self._current_stage == "base":
            from openclawmini.utils.llm_client import MistralClient
            api_key = os.getenv("MISTRAL_API_KEY", "")
            client = MistralClient(api_key=api_key, model=self.config.base_model.model)
            return client.complete

        # Fine-tuned stage: use ART's openai_client for inference
        import art  # type: ignore[import]

        model_name = self._current_model_name
        model = art.TrainableModel(
            name=model_name,
            project="openclawmini",
            base_model=model_name,
        )
        openai_client = model.openai_client()

        def art_model_fn(prompt: str) -> str:
            async def _call():
                resp = await openai_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    max_tokens=300,
                    temperature=0.3,
                )
                return resp.choices[0].message.content or ""
            return asyncio.run(_call())

        return art_model_fn

    def _build_initial_message(self, user_name: str, target: float) -> str:
        history = self._state.get("eval_history", [])
        training = self._state.get("training_history", [])
        spent = self._state.get("cost_usd", 0.0)

        lines = [
            f"Start autonomous fine-tuning for {user_name}.",
            f"Target: {target:.0%} overall accuracy",
            f"Budget: ${self._budget:.0f} total | ${spent:.2f} spent | "
            f"${self._budget - spent:.2f} remaining",
        ]

        if history:
            lines.append(f"\nEval history ({len(history)} runs):")
            for h in history[-3:]:
                lines.append(
                    f"  [{h['stage']}] "
                    f"factual={h.get('factual_accuracy', 0):.0%} "
                    f"stylistic={h.get('stylistic_accuracy', 0):.0%} "
                    f"overall={h.get('overall_accuracy', 0):.0%}"
                )
        else:
            lines.append("\nNo eval history — start with run_eval('base').")

        if training:
            lines.append(f"\nTraining history ({len(training)} actions):")
            for t in training[-3:]:
                lines.append(f"  [{t['action']}] {t.get('model_name', '')} "
                              f"(~${t.get('cost_est', 0):.0f})")

        lines.append("\nBegin.")
        return "\n".join(lines)

    def _build_tool_definitions(self) -> list:
        return [
            {
                "name": "run_eval",
                "description": (
                    "Run factual + stylistic evaluation against the current model. "
                    "Returns accuracy scores including per-category factual breakdown."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "stage": {
                            "type": "string",
                            "description": "Stage label (e.g. 'base', 'sft', 'grpo', 'grpo_v2')",
                        },
                    },
                    "required": ["stage"],
                },
            },
            {
                "name": "run_sft",
                "description": (
                    "Run supervised fine-tuning on the latest SFT training data. "
                    "Improves factual knowledge. Estimated cost: ~$10-15."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sft_path": {
                            "type": "string",
                            "description": "Path to SFT JSONL file. Leave empty to use the latest auto-detected file.",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "run_grpo",
                "description": (
                    "Run GRPO reinforcement learning from the latest SFT model. "
                    "Improves stylistic alignment using RULER scoring. Estimated cost: ~$7-12."
                ),
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "targeted_research",
                "description": (
                    "Conduct targeted web research to gather more facts about a specific "
                    "fact category where the model is underperforming. "
                    "Estimated cost: ~$0.50."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "The fact category to research",
                            "enum": [
                                "work", "education", "skills", "location",
                                "personal", "interests", "achievements", "other",
                            ],
                        },
                        "additional_context": {
                            "type": "string",
                            "description": "Optional extra guidance for the research",
                        },
                    },
                    "required": ["category"],
                },
            },
            {
                "name": "generate_sft_data",
                "description": (
                    "Generate new SFT training data using the DataSimulator SDK. "
                    "Run this after targeted_research to create focused training data. "
                    "Cost: free (just Gemini API calls for generation)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "focus_categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fact categories to emphasize (e.g. ['work', 'education']). Empty = balanced.",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "check_status",
                "description": "Check current budget, training history, and latest eval results.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "stop",
                "description": (
                    "Stop the training pipeline. Use when: target accuracy reached, "
                    "budget is low (< $3 remaining), or further training is unlikely to help."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why you are stopping",
                        },
                        "success": {
                            "type": "boolean",
                            "description": "True if target accuracy was reached",
                        },
                    },
                    "required": ["reason", "success"],
                },
            },
        ]

    def _build_final_result(self) -> OrchestratorResult:
        history = self._state.get("eval_history", [])
        target = getattr(self.config.training, "final_target_accuracy", 0.80)
        latest = history[-1] if history else {}
        best = max(history, key=lambda h: h.get("overall_accuracy", 0.0)) if history else {}

        result = OrchestratorResult(
            rounds_completed=len(self._state.get("training_history", [])),
            final_overall_accuracy=latest.get("overall_accuracy", 0.0),
            final_factual_accuracy=latest.get("factual_accuracy", 0.0),
            final_stylistic_accuracy=latest.get("stylistic_accuracy", 0.0),
            best_stage=best.get("stage", "base"),
            cost_usd=self._state.get("cost_usd", 0.0),
            target_reached=latest.get("overall_accuracy", 0.0) >= target,
            stop_reason=self._stop_reason or "Completed",
            eval_history=history,
            final_model_name=self._current_model_name,
            final_project="openclawmini",
        )

        self._console.print(Panel(
            "\n".join(result.summary_lines()),
            title="[bold yellow]Orchestrator Complete[/bold yellow]",
            border_style="yellow",
        ))

        return result

    # ── State persistence ───────────────────────────────────────

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"cost_usd": 0.0, "eval_history": [], "training_history": []}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_env(cls, config, memory, store) -> "OrchestratorAgent":
        """Build with all sub-agents from environment variables."""
        from openclawmini.agents.evals import EvalsAgent
        from openclawmini.agents.data_cleansing import DataCleansingAgent
        from openclawmini.agents.sft_agent import SFTAgent
        from openclawmini.agents.research import ResearchAgent
        from openclawmini.integrations.gemini_extractor import GeminiExtractor

        extractor = GeminiExtractor.from_env()
        evals_agent = EvalsAgent(gemini_extractor=extractor)
        data_agent = DataCleansingAgent.from_config(config)
        sft_agent = SFTAgent.from_config(
            config, user_name=getattr(config.user, "name", "")
        )
        research_agent = ResearchAgent(
            store=store, memory=memory, llm_client=extractor
        )

        return cls(
            config=config,
            memory=memory,
            store=store,
            evals_agent=evals_agent,
            data_agent=data_agent,
            sft_agent=sft_agent,
            research_agent=research_agent,
        )

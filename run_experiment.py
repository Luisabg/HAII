from __future__ import annotations
from dotenv import load_dotenv

import csv
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from anthropic import Anthropic

# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_CSV = Path("prompts.csv")
OUTPUT_JSONL = OUTPUT_DIR / "generations.jsonl"

DEFAULT_SLEEP_SECONDS = 1.0
RANDOM_SEED = 42

# Use one fixed model per provider for a given run.
# Replace these with the exact model IDs you decide to use.
MODELS = {
    "openai": "gpt-5",
    "anthropic": "claude-sonnet-4-5",
    "xai": "grok-4.20-reasoning",
}

# Fixed instruction for all providers unless you intentionally vary it.
BASE_INSTRUCTION = (
    "You are an assistant helping revise workplace writing. "
    "Follow the user's instruction exactly. "
    "Do not add extra explanation unless asked."
)

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class TrialRow:
    trial_id: str
    scenario_id: str
    condition_input_style: str
    condition_rewrite_prompt: str
    original_text: str

@dataclass
class GenerationResult:
    trial_id: str
    scenario_id: str
    provider: str
    model: str
    condition_input_style: str
    condition_rewrite_prompt: str
    original_text: str
    full_prompt: str
    output_text: Optional[str]
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    timestamp_utc: float
    latency_seconds: float

# -----------------------------
# Prompt construction
# -----------------------------

def build_user_prompt(row: TrialRow) -> str:
    """
    Build the exact user prompt for the trial.
    Keep this deterministic.
    """
    return (
        f"{row.condition_rewrite_prompt}\n\n"
        "Preserve the original meaning and core request.\n\n"
        "TEXT:\n"
        f"{row.original_text}"
    )

# -----------------------------
# Provider clients
# -----------------------------

class OpenAIProvider:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """
        Stateless OpenAI request:
        - no previous_response_id
        - no conversation
        - store=False
        """
        response = self.client.responses.create(
            model=self.model,
            store=False,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text


class AnthropicProvider:
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """
        Claude Messages API call.
        Keep each trial as a fresh request.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_instruction,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        # Claude returns content blocks; pull text blocks together.
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()


class XAIProvider:
    def __init__(self, api_key: str, model: str):
        """
        xAI documents both its SDK and an OpenAI-compatible Responses API.
        This starter uses the OpenAI client against xAI's base URL for consistency.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = model

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """
        Stateless xAI request:
        - fresh request each time
        - no previous_response_id
        """
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text

# -----------------------------
# Utilities
# -----------------------------

def load_trials(csv_path: Path) -> List[TrialRow]:
    rows: List[TrialRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "trial_id",
            "scenario_id",
            "condition_input_style",
            "condition_rewrite_prompt",
            "original_text",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for r in reader:
            rows.append(
                TrialRow(
                    trial_id=r["trial_id"],
                    scenario_id=r["scenario_id"],
                    condition_input_style=r["condition_input_style"],
                    condition_rewrite_prompt=r["condition_rewrite_prompt"],
                    original_text=r["original_text"],
                )
            )
    return rows


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_providers() -> Dict[str, object]:
    providers = {}

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")

    if openai_key:
        providers["openai"] = OpenAIProvider(
            api_key=openai_key,
            model=MODELS["openai"],
        )

    if anthropic_key:
        providers["anthropic"] = AnthropicProvider(
            api_key=anthropic_key,
            model=MODELS["anthropic"],
        )

    if xai_key:
        providers["xai"] = XAIProvider(
            api_key=xai_key,
            model=MODELS["xai"],
        )

    if not providers:
        raise RuntimeError(
            "No provider API keys found. Set OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, and/or XAI_API_KEY."
        )

    return providers


def run_single_trial(
    provider_name: str,
    provider,
    row: TrialRow,
    system_instruction: str,
) -> GenerationResult:
    full_prompt = build_user_prompt(row)
    start = time.time()

    try:
        output_text = provider.generate(system_instruction, full_prompt)
        latency = time.time() - start
        return GenerationResult(
            trial_id=row.trial_id,
            scenario_id=row.scenario_id,
            provider=provider_name,
            model=provider.model,
            condition_input_style=row.condition_input_style,
            condition_rewrite_prompt=row.condition_rewrite_prompt,
            original_text=row.original_text,
            full_prompt=full_prompt,
            output_text=output_text,
            success=True,
            error_type=None,
            error_message=None,
            timestamp_utc=time.time(),
            latency_seconds=latency,
        )
    except Exception as e:
        latency = time.time() - start
        return GenerationResult(
            trial_id=row.trial_id,
            scenario_id=row.scenario_id,
            provider=provider_name,
            model=getattr(provider, "model", "unknown"),
            condition_input_style=row.condition_input_style,
            condition_rewrite_prompt=row.condition_rewrite_prompt,
            original_text=row.original_text,
            full_prompt=full_prompt,
            output_text=None,
            success=False,
            error_type=type(e).__name__,
            error_message=str(e),
            timestamp_utc=time.time(),
            latency_seconds=latency,
        )


def run_experiment() -> None:
    random.seed(RANDOM_SEED)

    trials = load_trials(INPUT_CSV)
    providers = make_providers()

    # Cross all trials with all available providers
    jobs = [(provider_name, row) for row in trials for provider_name in providers.keys()]
    random.shuffle(jobs)

    print(f"Loaded {len(trials)} trials")
    print(f"Running across providers: {list(providers.keys())}")
    print(f"Total jobs: {len(jobs)}")

    for i, (provider_name, row) in enumerate(jobs, start=1):
        provider = providers[provider_name]
        result = run_single_trial(
            provider_name=provider_name,
            provider=provider,
            row=row,
            system_instruction=BASE_INSTRUCTION,
        )
        append_jsonl(OUTPUT_JSONL, asdict(result))

        status = "OK" if result.success else "ERR"
        print(
            f"[{i}/{len(jobs)}] {status} "
            f"{provider_name} trial={row.trial_id} "
            f"latency={result.latency_seconds:.2f}s"
        )

        time.sleep(DEFAULT_SLEEP_SECONDS)


if __name__ == "__main__":
    run_experiment()
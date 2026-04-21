"""
Tinker API Experiment Runner
Collects utterance distributions from DeepSeek-V3.1 (via Tinker API) under three prompt framings:
  - second_person: direct production ("You are the speaker…")
  - first_person:  assistant framing ("You are a helpful assistant…")
  - third_person:  meta-pragmatic judge ("A speaker must choose…")

For both phenomena:
  1. Scalar Implicature (Frank & Goodman, 2012)
  2. Polite Speech (Yoon et al., 2020)

Usage:
    python experiments/run_experiments.py --n_samples 25


Set TINKER_API_KEY in a .env file in the project root.
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads TINKER_API_KEY from .env file

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise EnvironmentError("TINKER_API_KEY not set. Add it to your .env file.")

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_MODEL    = "deepseek-ai/DeepSeek-V3.1"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rsa_model import get_frank_goodman_stimuli, get_yoon_semantics

SCALAR_PROMPTS = {
    "second_person": """You are playing a communication game with a partner.
There are {n_objects} objects on the table:
{object_list}

You need your partner to pick: {target}
Your partner cannot see which object you mean.

You may say exactly ONE word from this list:
{utterance_list}

Important: some words on the list describe more than one object and would be ambiguous. Choose the word that applies ONLY to the target and to no other object on the table. If no word is unique to the target, choose the word that is most helpful.

Reply with ONLY that one word. No punctuation, no explanation.""",

    "first_person": """I am playing a referential communication game.
The objects on the table are:
{object_list}

I need my partner to pick: {target}
I must say exactly ONE word from:
{utterance_list}

I should avoid words that describe multiple objects, because those would confuse my partner. I want the word that most uniquely identifies the target.

Reply with ONLY the single best word from the list. No explanation.""",

    "third_person": """A speaker and listener are playing a communication game.
Objects on the table:
{object_list}

The speaker needs the listener to pick: {target}
The speaker must say exactly ONE word from:
{utterance_list}

The speaker should reason: does this word describe only the target, or does it also describe other objects on the table? A word that describes multiple objects is ambiguous and unhelpful. The speaker should choose the word that most uniquely identifies the target.

Reply with ONLY the single best word. No explanation.""",
}

POLITE_PROMPTS = {
    "second_person": """\
Someone asked you to evaluate their work. Their true performance is {state} out of 5 stars.
You want to be both honest and kind.

Choose ONE utterance from the following options:
{utterance_list}

Which do you say? Reply with ONLY the exact utterance text, nothing else.""",

    "first_person": """\
As a helpful assistant, I need to give feedback on someone's work. \
Their actual performance is {state}/5 stars. I want to balance honesty with kindness.

My options are:
{utterance_list}

Which should I say? Reply with ONLY the exact utterance text from the list, nothing else.""",

    "third_person": """\
A speaker must give feedback on someone's work. The true quality is {state} out of 5 stars.
The speaker wants to balance being truthful with being kind.

Available utterances:
{utterance_list}

Which utterance should the speaker choose? Reply with ONLY the exact utterance text, nothing else.""",
}


def format_object_list(objects):
    """Display objects as 'a blue square' instead of 'blue_square'."""
    formatted = []
    for o in objects:
        parts = o.replace('_', ' ').split()
        formatted.append(f"  - a {' '.join(parts)}")
    return "\n".join(formatted)

def format_utterance_list(utterances):
    return "\n".join(f"  - {u}" for u in utterances)

def query_model(client, prompt, model=TINKER_MODEL, temperature=1.0, max_retries=3):
    """Single query to DeepSeek via Tinker API. Returns stripped response string."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=30,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    [RETRY {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    [ERROR] {e}")
                return None

def match_utterance(response, utterances):
    """
    Fuzzy-match model response to closest valid utterance.
    Returns matched utterance string or None.
    """
    if response is None:
        return None
    resp_lower = response.lower().strip().strip('"').strip("'")
    for u in utterances:
        if resp_lower == u.lower():
            return u
    for u in utterances:
        if u.lower() in resp_lower or resp_lower in u.lower():
            return u
    return None

def run_scalar_experiment(client, n_samples=25, verbose=True):
    """
    Run scalar implicature experiment across all scenes and framings.
    Returns dict: {scene_name: {framing: {utterance: count}}}
    """
    scenes   = get_frank_goodman_stimuli()
    framings = ["second_person", "first_person", "third_person"]
    results  = {}

    for scene in scenes:
        scene_name = scene['name']
        raw_target = scene['objects'][scene['target_idx']].replace('_', ' ')
        target     = 'a ' + raw_target
        results[scene_name] = {}

        for framing in framings:
            counts = defaultdict(int)
            if verbose:
                print(f"  [{scene_name}] framing={framing}", flush=True)

            for i in range(n_samples):
                prompt = SCALAR_PROMPTS[framing].format(
                    n_objects=len(scene['objects']),
                    object_list=format_object_list(scene['objects']),
                    target=target,
                    utterance_list=format_utterance_list(scene['utterances']),
                )
                raw = query_model(client, prompt)
                matched = match_utterance(raw, scene['utterances'])
                if matched:
                    counts[matched] += 1
                else:
                    counts['__unmatched__'] += 1
                    if verbose:
                        print(f"    Unmatched response: '{raw}'")

                time.sleep(0.3)

            results[scene_name][framing] = dict(counts)
            if verbose:
                print(f"    Distribution: {dict(counts)}")

    return results


def run_polite_experiment(client, n_samples=25, verbose=True):
    """
    Run polite speech experiment across all states and framings.
    Returns dict: {state: {framing: {utterance: count}}}
    """
    states, utterances, _ = get_yoon_semantics()
    framings = ["second_person", "first_person", "third_person"]
    results  = {}

    for state in states:
        state_key = f"state_{state}"
        results[state_key] = {}

        for framing in framings:
            counts = defaultdict(int)
            if verbose:
                print(f"  [state={state}/5] framing={framing}", flush=True)

            for i in range(n_samples):
                prompt = POLITE_PROMPTS[framing].format(
                    state=state,
                    utterance_list=format_utterance_list(utterances),
                )
                raw = query_model(client, prompt)
                matched = match_utterance(raw, utterances)
                if matched:
                    counts[matched] += 1
                else:
                    counts['__unmatched__'] += 1
                    if verbose:
                        print(f"    Unmatched: '{raw}'")

                time.sleep(0.3)

            results[state_key][framing] = dict(counts)
            if verbose:
                print(f"    Distribution: {dict(counts)}")

    return results


def main():
    global TINKER_MODEL
    parser = argparse.ArgumentParser(description="Run RSA vs DeepSeek experiments via Tinker API")
    parser.add_argument("--model",     type=str, default=TINKER_MODEL, help="Model string (default: deepseek-ai/DeepSeek-V3.1)")
    parser.add_argument("--n_samples", type=int, default=25, help="Samples per condition (20-30 recommended)")
    parser.add_argument("--phenomena", type=str, default="both", choices=["scalar","polite","both"])
    parser.add_argument("--output_dir",type=str, default="data/")
    parser.add_argument("--verbose",   action="store_true", default=True)
    args = parser.parse_args()

    client = OpenAI(
        api_key=TINKER_API_KEY,
        base_url=TINKER_BASE_URL,
    )

    TINKER_MODEL = args.model

    print(f"Using model : {TINKER_MODEL}")
    print(f"Base URL    : {TINKER_BASE_URL}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.phenomena in ("scalar", "both"):
        print("\n=== Running SCALAR IMPLICATURE experiment ===")
        scalar_results = run_scalar_experiment(client, n_samples=args.n_samples, verbose=args.verbose)
        all_results["scalar_implicature"] = scalar_results
        out_path = output_dir / "scalar_results.json"
        with open(out_path, "w") as f:
            json.dump(scalar_results, f, indent=2)
        print(f"  Saved → {out_path}")

    if args.phenomena in ("polite", "both"):
        print("\n=== Running POLITE SPEECH experiment ===")
        polite_results = run_polite_experiment(client, n_samples=args.n_samples, verbose=args.verbose)
        all_results["polite_speech"] = polite_results
        out_path = output_dir / "polite_results.json"
        with open(out_path, "w") as f:
            json.dump(polite_results, f, indent=2)
        print(f"  Saved → {out_path}")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_dir}/all_results.json")

    if args.phenomena in ("scalar", "both"):
        print("\n=== Running SCALAR IMPLICATURE experiment ===")
        scalar_results = run_scalar_experiment(client, n_samples=args.n_samples, verbose=args.verbose)
        all_results["scalar_implicature"] = scalar_results
        out_path = output_dir / "scalar_results.json"
        with open(out_path, "w") as f:
            json.dump(scalar_results, f, indent=2)
        print(f"  Saved → {out_path}")

    if args.phenomena in ("polite", "both"):
        print("\n=== Running POLITE SPEECH experiment ===")
        polite_results = run_polite_experiment(client, n_samples=args.n_samples, verbose=args.verbose)
        all_results["polite_speech"] = polite_results
        out_path = output_dir / "polite_results.json"
        with open(out_path, "w") as f:
            json.dump(polite_results, f, indent=2)
        print(f"  Saved → {out_path}")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_dir}/all_results.json")


if __name__ == "__main__":
    main()

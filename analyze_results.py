"""
analyze_results.py
Computes JS divergence, KL divergence, and Spearman rank correlation
between RSA model predictions and DeepSeek-V3.1 empirical distributions.

Usage (run from project root):
    python analyze_results.py
    python analyze_results.py --data_dir data/ --output_dir output/
    python analyze_results.py --omega_i 0.7 --alpha 5.0
"""

import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.special import rel_entr

warnings.filterwarnings("ignore")

from rsa_model import (
    ScalarImplicatureRSA, PoliteSpeechRSA,
    get_frank_goodman_stimuli, get_yoon_semantics
)

def smooth(p, eps=1e-10):
    p = np.array(p, dtype=float) + eps
    return p / p.sum()

def kl_divergence(p, q):
    """KL(p || q)"""
    p, q = smooth(p), smooth(q)
    return float(np.sum(rel_entr(p, q)))

def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric)"""
    p, q = smooth(p), smooth(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))

def js_distance(p, q):
    """Square root of JS divergence — proper metric bounded 0–1"""
    return float(np.sqrt(js_divergence(p, q)))

def spearman(p, q):
    if len(p) < 3:
        return np.nan, np.nan
    rho, pval = stats.spearmanr(p, q)
    return float(rho), float(pval)

def counts_to_dist(counts, utterances):
    arr = np.array([counts.get(u, 0) for u in utterances], dtype=float)
    return arr / arr.sum() if arr.sum() > 0 else np.ones(len(utterances)) / len(utterances)


def get_scalar_rsa_distributions():
    scenes = get_frank_goodman_stimuli()
    out = {}
    for scene in scenes:
        rsa = ScalarImplicatureRSA(
            objects=scene['objects'], utterances=scene['utterances'],
            lexicon=scene['lexicon'], prior=scene['prior'],
        )
        out[scene['name']] = {
            'rsa_speaker': rsa.pragmatic_speaker()[scene['target_idx']],
            'utterances':  scene['utterances'],
        }
    return out

def get_polite_rsa_distributions(omega_i=0.5, omega_s=0.5, alpha=3.0):
    states, utterances, semantics = get_yoon_semantics()
    rsa = PoliteSpeechRSA(states=states, utterances=utterances,
        semantics=semantics, alpha=alpha, omega_i=omega_i, omega_s=omega_s)
    S1 = rsa.pragmatic_speaker()
    return {
        f"state_{state}": {'rsa_speaker': S1[i], 'utterances': utterances}
        for i, state in enumerate(states)
    }


def analyze_phenomenon(llm_results, rsa_dists, phenomenon_name):
    framings = ["second_person", "first_person", "third_person"]
    records  = []

    for condition_key, framing_data in llm_results.items():
        if condition_key not in rsa_dists:
            print(f"  [WARN] no RSA dist for: {condition_key}")
            continue

        utterances = rsa_dists[condition_key]['utterances']
        rsa_dist   = rsa_dists[condition_key]['rsa_speaker']

        for framing in framings:
            if framing not in framing_data:
                continue

            llm_dist = counts_to_dist(framing_data[framing], utterances)
            rho, p   = spearman(rsa_dist, llm_dist)

            records.append({
                'phenomenon':    phenomenon_name,
                'condition':     condition_key,
                'framing':       framing,
                'js_divergence': round(js_divergence(rsa_dist, llm_dist), 4),
                'js_distance':   round(js_distance(rsa_dist, llm_dist), 4),
                'kl_rsa_llm':    round(kl_divergence(rsa_dist, llm_dist), 4),
                'kl_llm_rsa':    round(kl_divergence(llm_dist, rsa_dist), 4),
                'spearman_rho':  round(rho, 4) if not np.isnan(rho) else np.nan,
                'spearman_p':    round(p, 4)   if not np.isnan(p)   else np.nan,
                'rsa_dist':      list(np.round(rsa_dist, 4)),
                'llm_dist':      list(np.round(llm_dist, 4)),
                'utterances':    utterances,
            })

    return pd.DataFrame(records)


def print_detailed(df):
    for _, row in df.iterrows():
        print(f"\n{'─'*60}")
        print(f"{row['phenomenon']} | {row['condition']} | {row['framing']}")
        print(f"  {'Utterance':<24} {'RSA':>7} {'LLM':>7}")
        for utt, r, l in zip(row['utterances'], row['rsa_dist'], row['llm_dist']):
            marker = " <--" if abs(r - l) > 0.3 else ""
            print(f"  {utt:<24} {r:>7.3f} {l:>7.3f}{marker}")
        print(f"  JS divergence : {row['js_divergence']:.4f}")
        print(f"  KL(RSA||LLM)  : {row['kl_rsa_llm']:.4f}")
        print(f"  KL(LLM||RSA)  : {row['kl_llm_rsa']:.4f}")
        print(f"  Spearman rho  : {row['spearman_rho']:.4f}  (p={row['spearman_p']:.4f})")


def summarize(df):
    cols = ['js_divergence', 'js_distance', 'kl_rsa_llm', 'kl_llm_rsa', 'spearman_rho']
    return df.groupby(['phenomenon', 'framing'])[cols].agg(['mean', 'std']).round(4)


def omega_sweep(polite_data, output_dir):
    import itertools
    states, utterances, semantics = get_yoon_semantics()
    framings = ["second_person", "first_person", "third_person"]
    omega_vals = np.linspace(0, 1, 11)
    alpha_vals = [1.0, 3.0, 5.0]
    sweep = []

    for omega_i, alpha in itertools.product(omega_vals, alpha_vals):
        rsa = PoliteSpeechRSA(states=states, utterances=utterances,
            semantics=semantics, alpha=alpha, omega_i=omega_i, omega_s=1-omega_i)
        S1 = rsa.pragmatic_speaker()
        jsds = []
        for i, state in enumerate(states):
            for framing in framings:
                counts   = polite_data.get(f"state_{state}", {}).get(framing, {})
                llm_dist = counts_to_dist(counts, utterances)
                jsds.append(js_divergence(S1[i], llm_dist))
        sweep.append({'omega_i': round(omega_i, 2), 'alpha': alpha, 'mean_js': np.mean(jsds)})

    df_sweep = pd.DataFrame(sweep)
    best = df_sweep.loc[df_sweep['mean_js'].idxmin()]
    print(f"\nBest-fitting RSA params (lowest mean JS):")
    print(f"  omega_i={best['omega_i']}, alpha={best['alpha']}, mean_JS={best['mean_js']:.4f}")

    df_sweep.to_csv(output_dir / "omega_sweep.csv", index=False)
    return df_sweep, best


def make_plots(df_all, polite_data, output_dir, best_omega_i, best_alpha):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots"); return

    framings = ["second_person", "first_person", "third_person"]
    colors   = {'second_person': '#4C72B0', 'first_person': '#DD8452', 'third_person': '#55A868'}
    flabels  = {'second_person': '2nd person', 'first_person': '1st person', 'third_person': '3rd person'}
    phenomena = df_all['phenomenon'].unique()

    fig, axes = plt.subplots(1, len(phenomena), figsize=(7 * len(phenomena), 5), sharey=True)
    if len(phenomena) == 1: axes = [axes]
    for ax, phen in zip(axes, phenomena):
        sub = df_all[df_all['phenomenon'] == phen]
        conditions = list(sub['condition'].unique())
        x = np.arange(len(conditions))
        for j, framing in enumerate(framings):
            vals = [sub[(sub['condition']==c)&(sub['framing']==framing)]['js_divergence'].values
                    for c in conditions]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (j-1)*0.25, vals, 0.25, label=flabels[framing],
                   color=colors[framing], alpha=0.85)
        short = [c.split('(')[1].rstrip(')') if '(' in c else c for c in conditions]
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=25, ha='right', fontsize=8)
        ax.set_title(phen.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('JS Divergence'); ax.set_ylim(0, 0.75)
        ax.legend(fontsize=9)
    fig.suptitle('RSA vs DeepSeek-V3.1: JS Divergence', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_js_divergence.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved fig1_js_divergence.png")

    states, utterances, semantics = get_yoon_semantics()
    rsa = PoliteSpeechRSA(states=states, utterances=utterances, semantics=semantics,
        alpha=best_alpha, omega_i=best_omega_i, omega_s=1-best_omega_i)
    S1 = rsa.pragmatic_speaker()
    short_utts = [u.replace("It's ", '') for u in utterances]

    fig, axes = plt.subplots(2, len(states), figsize=(4*len(states), 7))
    for i, state in enumerate(states):
        rsa_dist = S1[i]
        llm_avg  = np.mean([counts_to_dist(
            polite_data.get(f"state_{state}", {}).get(fr, {}), utterances)
            for fr in framings], axis=0)
        for row, (dist, title, color) in enumerate([
            (rsa_dist, f'RSA S1\nstate={state}', '#4C72B0'),
            (llm_avg,  f'DeepSeek (avg)\nstate={state}', '#DD8452'),
        ]):
            ax = axes[row][i]
            ax.bar(range(len(utterances)), dist, color=color, alpha=0.8, edgecolor='white')
            ax.set_xticks(range(len(utterances)))
            ax.set_xticklabels(short_utts, rotation=45, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=9)
            if i == 0: ax.set_ylabel('Probability', fontsize=9)
    fig.suptitle(f'Polite Speech: RSA (omega_i={best_omega_i}) vs DeepSeek-V3.1',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_polite_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved fig2_polite_distributions.png")

    import itertools
    omega_vals = np.linspace(0, 1, 11)
    alpha_vals = [1.0, 3.0, 5.0]
    sweep_results = []
    for omega_i, alpha in itertools.product(omega_vals, alpha_vals):
        rsa_ = PoliteSpeechRSA(states=states, utterances=utterances, semantics=semantics,
            alpha=alpha, omega_i=omega_i, omega_s=1-omega_i)
        S1_ = rsa_.pragmatic_speaker()
        jsds = [js_divergence(S1_[i], counts_to_dist(
            polite_data.get(f"state_{s}", {}).get(fr, {}), utterances))
            for i, s in enumerate(states) for fr in framings]
        sweep_results.append((omega_i, alpha, np.mean(jsds)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in alpha_vals:
        pts = [(o, j) for o, a, j in sweep_results if a == alpha]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker='o', label=f'alpha={alpha}', linewidth=2)
    ax.axvline(best_omega_i, color='red', linestyle='--', alpha=0.6)
    ax.set_xlabel('omega_i (informational weight)', fontsize=11)
    ax.set_ylabel('Mean JS Divergence', fontsize=11)
    ax.set_title('RSA Parameter Sweep: Best Fit for DeepSeek Polite Speech',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_omega_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved fig3_omega_sweep.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze RSA vs DeepSeek results")
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--omega_i",    type=float, default=0.5)
    parser.add_argument("--omega_s",    type=float, default=0.5)
    parser.add_argument("--alpha",      type=float, default=3.0)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    scalar_path = data_dir / "scalar_results.json"
    if scalar_path.exists():
        print("\n=== Scalar Implicature ===")
        with open(scalar_path) as f: scalar_data = json.load(f)
        rsa_scalar = get_scalar_rsa_distributions()
        df_scalar  = analyze_phenomenon(scalar_data, rsa_scalar, "scalar_implicature")
        all_dfs.append(df_scalar)
        print_detailed(df_scalar)
    else:
        print(f"[SKIP] {scalar_path} not found")

    polite_path = data_dir / "polite_results.json"
    if polite_path.exists():
        print("\n=== Polite Speech ===")
        with open(polite_path) as f: polite_data = json.load(f)
        rsa_polite = get_polite_rsa_distributions(
            omega_i=args.omega_i, omega_s=args.omega_s, alpha=args.alpha)
        df_polite  = analyze_phenomenon(polite_data, rsa_polite, "polite_speech")
        all_dfs.append(df_polite)
        print_detailed(df_polite)
    else:
        print(f"[SKIP] {polite_path} not found")
        polite_data = {}

    if not all_dfs:
        print("\nNo data found. Run run_experiments.py first."); return

    df_all = pd.concat(all_dfs, ignore_index=True)

    df_csv = df_all.drop(columns=['rsa_dist', 'llm_dist', 'utterances'], errors='ignore')
    df_csv.to_csv(output_dir / "full_results.csv", index=False)
    print(f"\nSaved full_results.csv")

    print("\n=== SUMMARY (mean across conditions) ===")
    summary = summarize(df_all)
    print(summary.to_string())
    summary.to_csv(output_dir / "summary.csv")

    if polite_data:
        print("\n=== Omega/Alpha Sweep ===")
        _, best = omega_sweep(polite_data, output_dir)
        best_omega_i = best['omega_i']
        best_alpha   = best['alpha']
    else:
        best_omega_i, best_alpha = args.omega_i, args.alpha

    print("\n=== Generating plots ===")
    make_plots(df_all, polite_data, output_dir, best_omega_i, best_alpha)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()

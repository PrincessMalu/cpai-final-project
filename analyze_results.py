"""
analyze_results.py
Computes JS divergence, KL divergence, and Spearman rank correlation
between RSA model predictions and LLM empirical distributions.

Usage (run from project root):
    python analyze_results.py
    python analyze_results.py --data_dir data/ --output_dir output/
    python analyze_results.py --omega_i 0.7 --alpha 5.0
"""

import json
import re
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

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.1"

def model_slug(model_name):
    """Convert a model identifier into a filesystem-safe directory name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    return slug or "model"

def load_results_file(path):
    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload:
        return payload["results"], payload.get("metadata", {})
    return payload, {}

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

def counts_to_array(counts, utterances):
    return np.array([counts.get(u, 0) for u in utterances], dtype=int)

def bootstrap_dist_and_js(counts, utterances, rsa_dist, n_boot=2000, ci=95, rng=None):
    """
    Bootstrap the empirical utterance distribution and its JS divergence to RSA.
    Resamples utterance draws with replacement from the observed count distribution.
    """
    rng = np.random.default_rng() if rng is None else rng
    count_arr = counts_to_array(counts, utterances)
    n = int(count_arr.sum())
    k = len(utterances)
    alpha = (100 - ci) / 2

    point_dist = counts_to_dist(counts, utterances)
    point_js = js_divergence(rsa_dist, point_dist)

    if n == 0:
        uniform = np.ones(k) / k
        return {
            'n_matched': 0,
            'llm_dist_boot_mean': uniform,
            'llm_dist_ci_low': uniform,
            'llm_dist_ci_high': uniform,
            'js_boot_mean': point_js,
            'js_ci_low': point_js,
            'js_ci_high': point_js,
            'js_boot_std': 0.0,
        }

    probs = count_arr / n
    boot_counts = rng.multinomial(n, probs, size=n_boot)
    boot_dists = boot_counts / n
    boot_js = np.array([js_divergence(rsa_dist, dist) for dist in boot_dists], dtype=float)

    return {
        'n_matched': n,
        'llm_dist_boot_mean': boot_dists.mean(axis=0),
        'llm_dist_ci_low': np.percentile(boot_dists, alpha, axis=0),
        'llm_dist_ci_high': np.percentile(boot_dists, 100 - alpha, axis=0),
        'js_boot_mean': float(boot_js.mean()),
        'js_ci_low': float(np.percentile(boot_js, alpha)),
        'js_ci_high': float(np.percentile(boot_js, 100 - alpha)),
        'js_boot_std': float(boot_js.std(ddof=1)) if len(boot_js) > 1 else 0.0,
    }

def subsample_js_stability(counts, utterances, rsa_dist, fractions=(0.5, 0.7, 0.9), n_resamples=1000, rng=None):
    """
    Evaluate how stable JS divergence is under smaller without-replacement subsamples
    of the observed matched responses.
    """
    rng = np.random.default_rng() if rng is None else rng
    count_arr = counts_to_array(counts, utterances)
    n = int(count_arr.sum())
    rows = []

    if n == 0:
        return rows

    labels = np.repeat(np.arange(len(utterances)), count_arr)
    full_dist = count_arr / n
    full_js = js_divergence(rsa_dist, full_dist)

    for frac in fractions:
        sub_n = max(1, int(round(n * frac)))
        js_vals = np.empty(n_resamples, dtype=float)
        for i in range(n_resamples):
            sample = rng.choice(labels, size=sub_n, replace=False)
            sub_counts = np.bincount(sample, minlength=len(utterances))
            sub_dist = sub_counts / sub_n
            js_vals[i] = js_divergence(rsa_dist, sub_dist)

        rows.append({
            'subsample_fraction': frac,
            'subsample_n': sub_n,
            'full_sample_n': n,
            'full_sample_js': full_js,
            'js_subsample_mean': float(js_vals.mean()),
            'js_subsample_std': float(js_vals.std(ddof=1)) if len(js_vals) > 1 else 0.0,
            'js_subsample_ci_low': float(np.percentile(js_vals, 2.5)),
            'js_subsample_ci_high': float(np.percentile(js_vals, 97.5)),
            'js_delta_mean_abs': float(np.mean(np.abs(js_vals - full_js))),
            'js_delta_max_abs': float(np.max(np.abs(js_vals - full_js))),
        })

    return rows


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


def analyze_phenomenon(llm_results, rsa_dists, phenomenon_name, model_name, n_boot=2000, n_subsamples=1000, seed=7):
    framings = ["second_person", "first_person", "third_person"]
    records  = []
    stability_records = []
    base_rng = np.random.default_rng(seed)

    for condition_key, framing_data in llm_results.items():
        if condition_key not in rsa_dists:
            print(f"  [WARN] no RSA dist for: {condition_key}")
            continue

        utterances = rsa_dists[condition_key]['utterances']
        rsa_dist   = rsa_dists[condition_key]['rsa_speaker']

        for framing in framings:
            if framing not in framing_data:
                continue

            counts = framing_data[framing]
            llm_dist = counts_to_dist(counts, utterances)
            rho, p   = spearman(rsa_dist, llm_dist)
            boot = bootstrap_dist_and_js(
                counts, utterances, rsa_dist,
                n_boot=n_boot,
                rng=np.random.default_rng(base_rng.integers(0, 2**32 - 1)),
            )
            stability_rows = subsample_js_stability(
                counts, utterances, rsa_dist,
                n_resamples=n_subsamples,
                rng=np.random.default_rng(base_rng.integers(0, 2**32 - 1)),
            )
            for row in stability_rows:
                stability_records.append({
                    'model': model_name,
                    'phenomenon': phenomenon_name,
                    'condition': condition_key,
                    'framing': framing,
                    **row,
                })

            records.append({
                'model':         model_name,
                'phenomenon':    phenomenon_name,
                'condition':     condition_key,
                'framing':       framing,
                'js_divergence': round(js_divergence(rsa_dist, llm_dist), 4),
                'js_distance':   round(js_distance(rsa_dist, llm_dist), 4),
                'kl_rsa_llm':    round(kl_divergence(rsa_dist, llm_dist), 4),
                'kl_llm_rsa':    round(kl_divergence(llm_dist, rsa_dist), 4),
                'spearman_rho':  round(rho, 4) if not np.isnan(rho) else np.nan,
                'spearman_p':    round(p, 4)   if not np.isnan(p)   else np.nan,
                'n_matched':     boot['n_matched'],
                'js_boot_mean':  round(boot['js_boot_mean'], 4),
                'js_ci_low':     round(boot['js_ci_low'], 4),
                'js_ci_high':    round(boot['js_ci_high'], 4),
                'js_boot_std':   round(boot['js_boot_std'], 4),
                'rsa_dist':      list(np.round(rsa_dist, 4)),
                'llm_dist':      list(np.round(llm_dist, 4)),
                'llm_ci_low':    list(np.round(boot['llm_dist_ci_low'], 4)),
                'llm_ci_high':   list(np.round(boot['llm_dist_ci_high'], 4)),
                'utterances':    utterances,
            })

    return pd.DataFrame(records), pd.DataFrame(stability_records)


def print_detailed(df):
    for _, row in df.iterrows():
        print(f"\n{'─'*60}")
        print(f"{row['phenomenon']} | {row['condition']} | {row['framing']}")
        print(f"  {'Utterance':<24} {'RSA':>7} {'LLM':>7} {'95% CI':>17}")
        for utt, r, l, lo, hi in zip(row['utterances'], row['rsa_dist'], row['llm_dist'],
                                     row['llm_ci_low'], row['llm_ci_high']):
            marker = " <--" if abs(r - l) > 0.3 else ""
            print(f"  {utt:<24} {r:>7.3f} {l:>7.3f} [{lo:0.3f}, {hi:0.3f}]{marker}")
        print(f"  JS divergence : {row['js_divergence']:.4f}  (95% bootstrap CI [{row['js_ci_low']:.4f}, {row['js_ci_high']:.4f}])")
        print(f"  KL(RSA||LLM)  : {row['kl_rsa_llm']:.4f}")
        print(f"  KL(LLM||RSA)  : {row['kl_llm_rsa']:.4f}")
        print(f"  Spearman rho  : {row['spearman_rho']:.4f}  (p={row['spearman_p']:.4f})")

def summarize_stability(df_stability):
    if df_stability.empty:
        return pd.DataFrame()

    cols = ['full_sample_js', 'js_subsample_mean', 'js_subsample_std',
            'js_delta_mean_abs', 'js_delta_max_abs']
    return df_stability.groupby(
        ['model', 'phenomenon', 'framing', 'subsample_fraction']
    )[cols].agg(['mean', 'std']).round(4)


def summarize(df):
    cols = ['js_divergence', 'js_ci_low', 'js_ci_high', 'js_distance',
            'kl_rsa_llm', 'kl_llm_rsa', 'spearman_rho']
    return df.groupby(['model', 'phenomenon', 'framing'])[cols].agg(['mean', 'std']).round(4)


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


def display_model_name(model_name):
    return model_name.split("/")[-1]


def make_plots(df_all, polite_data, output_dir, best_omega_i, best_alpha, model_name):
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
    plot_model_name = display_model_name(model_name)

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
    fig.suptitle(f'RSA vs {plot_model_name}: JS Divergence', fontsize=13, fontweight='bold')
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
            (llm_avg,  f'{plot_model_name} (avg)\nstate={state}', '#DD8452'),
        ]):
            ax = axes[row][i]
            ax.bar(range(len(utterances)), dist, color=color, alpha=0.8, edgecolor='white')
            ax.set_xticks(range(len(utterances)))
            ax.set_xticklabels(short_utts, rotation=45, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=9)
            if i == 0: ax.set_ylabel('Probability', fontsize=9)
    fig.suptitle(f'Polite Speech: RSA (omega_i={best_omega_i}) vs {plot_model_name}',
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
    ax.set_title(f'RSA Parameter Sweep: Best Fit for {plot_model_name} Polite Speech',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_omega_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved fig3_omega_sweep.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze RSA vs DeepSeek results")
    parser.add_argument("--model",      type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--omega_i",    type=float, default=0.5)
    parser.add_argument("--omega_s",    type=float, default=0.5)
    parser.add_argument("--alpha",      type=float, default=3.0)
    parser.add_argument("--n_boot",     type=int, default=2000)
    parser.add_argument("--n_subsamples", type=int, default=1000)
    parser.add_argument("--seed",       type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / model_slug(args.model)
    output_dir = Path(args.output_dir) / model_slug(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {args.model}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")

    all_dfs = []
    stability_dfs = []

    scalar_path = data_dir / "scalar_results.json"
    if scalar_path.exists():
        print("\n=== Scalar Implicature ===")
        scalar_data, scalar_meta = load_results_file(scalar_path)
        scalar_model_name = scalar_meta.get("model", args.model)
        if scalar_meta:
            print(f"Metadata   : {scalar_meta}")
        rsa_scalar = get_scalar_rsa_distributions()
        df_scalar, df_scalar_stability = analyze_phenomenon(
            scalar_data, rsa_scalar, "scalar_implicature", scalar_model_name,
            n_boot=args.n_boot, n_subsamples=args.n_subsamples, seed=args.seed,
        )
        all_dfs.append(df_scalar)
        stability_dfs.append(df_scalar_stability)
        print_detailed(df_scalar)
    else:
        print(f"[SKIP] {scalar_path} not found")

    polite_path = data_dir / "polite_results.json"
    if polite_path.exists():
        print("\n=== Polite Speech ===")
        polite_data, polite_meta = load_results_file(polite_path)
        polite_model_name = polite_meta.get("model", args.model)
        if polite_meta:
            print(f"Metadata   : {polite_meta}")
        rsa_polite = get_polite_rsa_distributions(
            omega_i=args.omega_i, omega_s=args.omega_s, alpha=args.alpha)
        df_polite, df_polite_stability = analyze_phenomenon(
            polite_data, rsa_polite, "polite_speech", polite_model_name,
            n_boot=args.n_boot, n_subsamples=args.n_subsamples, seed=args.seed + 1,
        )
        all_dfs.append(df_polite)
        stability_dfs.append(df_polite_stability)
        print_detailed(df_polite)
    else:
        print(f"[SKIP] {polite_path} not found")
        polite_data = {}

    if not all_dfs:
        print("\nNo data found. Run run_experiments.py first."); return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_stability = pd.concat(stability_dfs, ignore_index=True) if stability_dfs else pd.DataFrame()

    df_csv = df_all.drop(columns=['rsa_dist', 'llm_dist', 'utterances'], errors='ignore')
    df_csv.to_csv(output_dir / "full_results.csv", index=False)
    print(f"\nSaved full_results.csv")

    print("\n=== SUMMARY (mean across conditions) ===")
    summary = summarize(df_all)
    print(summary.to_string())
    summary.to_csv(output_dir / "summary.csv")

    if not df_stability.empty:
        df_stability.to_csv(output_dir / "js_stability.csv", index=False)
        print("\n=== JS STABILITY ACROSS SUBSAMPLES ===")
        stability_summary = summarize_stability(df_stability)
        print(stability_summary.to_string())
        stability_summary.to_csv(output_dir / "js_stability_summary.csv")
        print("\nSaved js_stability.csv")
        print("Saved js_stability_summary.csv")

    if polite_data:
        print("\n=== Omega/Alpha Sweep ===")
        _, best = omega_sweep(polite_data, output_dir)
        best_omega_i = best['omega_i']
        best_alpha   = best['alpha']
    else:
        best_omega_i, best_alpha = args.omega_i, args.alpha

    print("\n=== Generating plots ===")
    make_plots(df_all, polite_data, output_dir, best_omega_i, best_alpha, args.model)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()

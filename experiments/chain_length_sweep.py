"""Experiment: how fast does invariance decay as the transformation chain grows?

The central claim of this project is that failures accumulate along a
multi-step transformation chain in ways single-step perturbations miss. This
script makes that quantitative: it builds one long chain s0 -> s1 -> ... -> sL
per seed sentence, then measures, for each prefix length k, the fraction of
sentences whose predicted label is unchanged from s0 to sk (an invariance
pass-rate). A monotone decline in that curve is exactly the "cumulative drift"
the report describes.

Run from the repository root:

    python -m experiments.chain_length_sweep

It reuses the validated building blocks in ``metamorphic_testing.py`` and only
adds the sweep + aggregation on top, so its numerics match the main pipeline.
"""

import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metamorphic_testing import (  # noqa: E402
    PERTURB_STEPS_GRAMMAR,
    _pred_label,
    build_chains,
    load_seed_sentences,
    make_predictor,
)
from fuzzingbook.GrammarFuzzer import GrammarFuzzer  # noqa: E402

CHAIN_LENGTH = 12
NUM_SEEDS = 100


def sample_fixed_length_pipeline(length):
    """Sample a perturbation pipeline of a fixed number of steps."""
    fuzzer = GrammarFuzzer(
        PERTURB_STEPS_GRAMMAR, min_nonterminals=length, max_nonterminals=4 * length
    )
    ops = fuzzer.fuzz().split()
    # Trim or pad (by resampling) to the exact requested length.
    while len(ops) < length:
        ops += fuzzer.fuzz().split()
    return ops[:length]


def invariance_pass_rate(chains, hf_softmax, k, batch_size=64):
    """Fraction of seeds whose predicted label is unchanged from s0 to sk."""
    n = len(chains)
    unchanged = 0
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        orig = hf_softmax([chains[i][0] for i in range(start, end)])
        pert = hf_softmax([chains[i][k] for i in range(start, end)])
        for op, pp in zip(orig, pert):
            if _pred_label(pp) == _pred_label(op):
                unchanged += 1
        start = end
    return unchanged / n


def main():
    sentences = load_seed_sentences(n=NUM_SEEDS)
    nlp = spacy.load("en_core_web_sm")

    ops = sample_fixed_length_pipeline(CHAIN_LENGTH)
    print("Pipeline:", ops)

    chains = build_chains(nlp, sentences, ops)
    _, hf_softmax = make_predictor()

    print(f"\n{'k':>3} {'operator applied':>20} {'invariance pass-rate':>22}")
    rates = []
    for k in range(1, len(ops) + 1):
        rate = invariance_pass_rate(chains, hf_softmax, k)
        rates.append(rate)
        print(f"{k:>3} {ops[k-1]:>20} {rate:>21.1%}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(ops) + 1), rates, "o-")
        plt.xlabel("Chain length k")
        plt.ylabel("Invariance pass-rate (label unchanged vs s0)")
        plt.title("Invariance decays as the transformation chain grows")
        plt.grid(True)
        plt.tight_layout()
        out = Path(__file__).resolve().parent / "chain_length_sweep.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved plot to {out}")
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()

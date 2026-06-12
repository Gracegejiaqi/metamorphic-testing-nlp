"""Grammar-based multi-step metamorphic testing for NLP models.

Extends the CheckList behavioral-testing workflow in two ways:

1. Multi-step transformation chains: each seed sentence s0 is expanded into
   a chain s0 -> s1 -> ... -> sL by composing CheckList perturbation
   operators, exposing cumulative and interaction effects that single-step
   perturbations miss.
2. Grammar-based fuzzing: both the perturbation pipeline and the metamorphic
   property to check are sampled from grammars (via The Fuzzing Book's
   GrammarFuzzer), automating test generation beyond manual design.

See report/report.pdf for the full write-up.
"""

import re

import pandas as pd
import spacy
import torch
from checklist.expect import Expect
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import DIR, INV
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from transformers import pipeline

DATA_URL = "hf://datasets/osanseviero/twitter-airline-sentiment/Tweets.csv"
NUM_SEEDS = 100

PERTURB_STEPS_GRAMMAR = {
    "<start>": ["<steps>"],
    "<steps>": ["<step>", "<step> <steps>"],
    "<step>": [
        "add_typos",
        "punctuation",
        "strip_punctuation",
        "contractions",
        "change_names",
        "change_location",
        "change_number",
        "add_negation",
        "remove_negation",
    ],
}

RE_INV = re.compile(r"^INV_FINAL\((\d+)\)$")
RE_DIR = re.compile(r"^DIR_FINAL\((\d+),EXPECT=([A-Z_]+)\)$")
RE_DIR_NAME = re.compile(r"^DIR final-only k=(\d+), expect=([A-Z_]+)")
RE_INV_NAME = re.compile(r"^INV final-only k=(\d+)")


def load_seed_sentences(url=DATA_URL, n=NUM_SEEDS):
    """Load the first n tweet texts from the Twitter US Airline Sentiment dataset."""
    df = pd.read_csv(url)
    sentences = df["text"].astype(str).tolist()[:n]
    print("Loaded examples:", sentences[:5])
    return sentences


def sample_pipeline():
    """Sample a variable-length perturbation pipeline from the steps grammar."""
    seq_fuzzer = GrammarFuzzer(
        PERTURB_STEPS_GRAMMAR, min_nonterminals=3, max_nonterminals=50
    )
    ops = seq_fuzzer.fuzz().split()
    print("Random pipeline steps:", ops)
    return ops


def sample_property(chain_length):
    """Sample one metamorphic property (INV or DIR on the final chain index)."""
    property_grammar = {
        "<start>": ["<prop>"],
        "<prop>": [
            "INV_FINAL(<k>)",
            "DIR_FINAL(<k>,EXPECT=TO_NEG_IF_POS)",
            "DIR_FINAL(<k>,EXPECT=TO_POS_IF_NEG)",
            "DIR_FINAL(<k>,EXPECT=MORE_NEGATIVE)",
            "DIR_FINAL(<k>,EXPECT=MORE_POSITIVE)",
            "DIR_FINAL(<k>,EXPECT=CHANGE_LABEL)",
        ],
        "<k>": [str(chain_length)],
    }
    prop_fuzzer = GrammarFuzzer(property_grammar, min_nonterminals=1, max_nonterminals=5)
    prop = prop_fuzzer.fuzz().strip()
    print("\nGrammar-picked property:", prop)
    return prop


def apply_one_op(nlp, s, token):
    """Apply a single CheckList perturbation to sentence s.

    Returns the first available transformed output, falling back to the
    original sentence if the transformation fails or produces nothing.
    """
    if token == "add_typos":
        r = Perturb.perturb([s], Perturb.add_typos)
    elif token == "contractions":
        r = Perturb.perturb([s], Perturb.contractions)
    elif token == "punctuation":
        r = Perturb.perturb([nlp(s)], Perturb.punctuation)
    elif token == "strip_punctuation":
        r = Perturb.perturb([nlp(s)], Perturb.strip_punctuation)
    elif token == "change_names":
        r = Perturb.perturb([nlp(s)], Perturb.change_names, n=3)
    elif token == "change_location":
        r = Perturb.perturb([nlp(s)], Perturb.change_location, n=3)
    elif token == "change_number":
        r = Perturb.perturb([nlp(s)], Perturb.change_number, n=3)
    elif token == "add_negation":
        r = Perturb.perturb([nlp(s)], Perturb.add_negation)
    elif token == "remove_negation":
        try:
            new_s = Perturb.remove_negation(nlp(s))
            return new_s if new_s and new_s != s else s
        except Exception:
            return s
    else:
        return s

    if r is None or not hasattr(r, "data") or not r.data:
        return s
    new_list = r.data[0][1:]
    if not new_list:
        return s
    out = new_list[0]
    return out if out else s


def build_chains(nlp, sentences, ops):
    """Expand each seed sentence into a transformation chain by applying ops in order."""
    chains = [[s] for s in sentences]
    for step_idx, op in enumerate(ops, start=1):
        print(f"\nApplying step S{step_idx}: {op}")
        num_changed = 0
        for chain in chains:
            current = chain[-1]
            new = apply_one_op(nlp, current, op)
            if new != current:
                num_changed += 1
            chain.append(new)
        print(f"S{step_idx}: changed {num_changed} / {len(chains)} samples")
    return chains


def make_predictor():
    """Wrap a HuggingFace sentiment pipeline as a CheckList softmax predictor.

    Returns (wrapped_predictor, raw_softmax_fn). Probabilities are ordered
    [negative, positive].
    """
    device = 0 if torch.backends.mps.is_available() else -1
    clf = pipeline("sentiment-analysis", device=device)

    def hf_softmax(xs):
        texts = [x[-1] if isinstance(x, (tuple, list)) else x for x in xs]
        outputs = clf(texts, truncation=True)
        probs = []
        for o in outputs:
            label = o["label"].upper()
            score = float(o["score"])
            if label.startswith("NEG"):
                probs.append([score, 1.0 - score])
            else:
                probs.append([1.0 - score, score])
        return probs

    return PredictorWrapper.wrap_softmax(hf_softmax), hf_softmax


def build_dir_expect(expect_name):
    """Map a DIR expectation token to a CheckList expectation function."""
    eps = 1e-12

    if expect_name == "TO_NEG_IF_POS":
        e = Expect.eq(0)
        return Expect.slice_orig(e, lambda orig, *args: orig == 1)

    if expect_name == "TO_POS_IF_NEG":
        e = Expect.eq(1)
        return Expect.slice_orig(e, lambda orig, *args: orig == 0)

    if expect_name == "MORE_NEGATIVE":
        def more_neg(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
            try:
                return float(conf[0]) > float(orig_conf[0]) + eps
            except Exception:
                return None
        return Expect.pairwise(more_neg)

    if expect_name == "MORE_POSITIVE":
        def more_pos(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
            try:
                return float(conf[1]) > float(orig_conf[1]) + eps
            except Exception:
                return None
        return Expect.pairwise(more_pos)

    if expect_name == "CHANGE_LABEL":
        def changed_pred(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
            return pred != orig_pred
        return Expect.pairwise(changed_pred)

    raise ValueError(f"Unknown DIR expectation: {expect_name}")


def build_tests(chains, props):
    """Instantiate CheckList INV/DIR tests from grammar-sampled properties."""
    tests = []
    for p in props:
        m = RE_INV.match(p)
        if m:
            k = int(m.group(1))
            data_k = [[c[0], c[k]] for c in chains]
            tests.append(INV(
                data=data_k,
                name=f"INV final-only k={k} (grammar)",
                capability="Robustness",
            ))
            continue

        m = RE_DIR.match(p)
        if m:
            k = int(m.group(1))
            expect_name = m.group(2)
            data_k = [[c[0], c[k]] for c in chains]
            tests.append(DIR(
                data=data_k,
                expect=build_dir_expect(expect_name),
                name=f"DIR final-only k={k}, expect={expect_name} (grammar)",
                capability="Robustness",
                description=(
                    "Metamorphic property generated by grammar; "
                    "results are for user interpretation."
                ),
            ))
            continue

    print(f"\nBuilt {len(tests)} tests: grammar-picked INV/DIR only.")
    return tests


def _pred_label(prob):
    return 0 if prob[0] >= prob[1] else 1


def _lab(x):
    return "NEG" if x == 0 else "POS"


def _fmt(prob):
    return f"{prob[0]:.4f},{prob[1]:.4f}({_lab(_pred_label(prob))})"


def _dir_considered_and_pass(expect_name, orig_prob, prob, eps=1e-12):
    """Return (considered, passed) for a DIR expectation on one example pair."""
    orig_pred = _pred_label(orig_prob)
    pred = _pred_label(prob)

    if expect_name == "TO_NEG_IF_POS":
        return (orig_pred == 1), (pred == 0)
    if expect_name == "TO_POS_IF_NEG":
        return (orig_pred == 0), (pred == 1)
    if expect_name == "MORE_NEGATIVE":
        try:
            return True, (float(prob[0]) > float(orig_prob[0]) + eps)
        except Exception:
            return True, False
    if expect_name == "MORE_POSITIVE":
        try:
            return True, (float(prob[1]) > float(orig_prob[1]) + eps)
        except Exception:
            return True, False
    if expect_name == "CHANGE_LABEL":
        return True, (pred != orig_pred)
    return False, False


def _collect_examples(chains, hf_softmax, k, judge, n_fail, n_pass, batch_size):
    """Run the model over (s0, sk) pairs and bucket them into passes/fails.

    judge(orig_prob, prob) -> (considered, passed)
    """
    fails, passes = [], []
    filtered = 0
    n = len(chains)
    start = 0
    while start < n and (len(fails) < n_fail or len(passes) < n_pass):
        end = min(start + batch_size, n)
        orig_texts = [chains[i][0] for i in range(start, end)]
        pert_texts = [chains[i][k] for i in range(start, end)]
        orig_probs = hf_softmax(orig_texts)
        pert_probs = hf_softmax(pert_texts)

        for op, pp, s0, sk in zip(orig_probs, pert_probs, orig_texts, pert_texts):
            considered, passed = judge(op, pp)
            if not considered:
                filtered += 1
                continue
            rec = (op, pp, s0, sk)
            if passed and len(passes) < n_pass:
                passes.append(rec)
            if not passed and len(fails) < n_fail:
                fails.append(rec)
            if len(fails) >= n_fail and len(passes) >= n_pass:
                break
        start = end
    return fails, passes, filtered


def _print_examples(fails, passes, n_fail, n_pass):
    print(f"\nExample fails (showing {len(fails)}/{n_fail}):")
    for op, pp, s0, sk in fails:
        print(f"{_fmt(op)} {s0}")
        print(f"{_fmt(pp)} {sk}")
        print("\n----")

    print(f"\nExample passes (showing {len(passes)}/{n_pass}):")
    for op, pp, s0, sk in passes:
        print(f"{_fmt(op)} {s0}")
        print(f"{_fmt(pp)} {sk}")
        print("\n----")


def print_dir_examples(chains, hf_softmax, test_name, n_fail=10, n_pass=10, batch_size=64):
    """Print representative passing/failing pairs for a DIR test."""
    m = RE_DIR_NAME.match(test_name)
    if not m:
        return
    k = int(m.group(1))
    expect_name = m.group(2)

    def judge(op, pp):
        return _dir_considered_and_pass(expect_name, op, pp)

    fails, passes, filtered = _collect_examples(
        chains, hf_softmax, k, judge, n_fail, n_pass, batch_size
    )
    print(f"\nFiltered out (orig not applicable): {filtered}")
    _print_examples(fails, passes, n_fail, n_pass)


def print_inv_examples(chains, hf_softmax, test_name, n_fail=10, n_pass=10, batch_size=64):
    """Print representative passing/failing pairs for an INV test."""
    m = RE_INV_NAME.match(test_name)
    if not m:
        return
    k = int(m.group(1))

    def judge(op, pp):
        return True, (_pred_label(pp) == _pred_label(op))

    fails, passes, _ = _collect_examples(
        chains, hf_softmax, k, judge, n_fail, n_pass, batch_size
    )
    _print_examples(fails, passes, n_fail, n_pass)


def main():
    sentences = load_seed_sentences()
    nlp = spacy.load("en_core_web_sm")

    ops = sample_pipeline()
    chains = build_chains(nlp, sentences, ops)
    print(f"\nTotal steps in this pipeline: {len(ops)}")

    props = [sample_property(len(ops))]
    wrapped, hf_softmax = make_predictor()
    tests = build_tests(chains, props)

    for test in tests:
        print(f"\n===== {test.name} =====")
        test.run(wrapped)
        if test.name.startswith("DIR final-only"):
            print_dir_examples(chains, hf_softmax, test.name)
        elif test.name.startswith("INV final-only"):
            print_inv_examples(chains, hf_softmax, test.name)


if __name__ == "__main__":
    main()

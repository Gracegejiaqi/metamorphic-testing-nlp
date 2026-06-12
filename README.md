# Grammar-based Multi-step Metamorphic Testing for NLP Models

Behavioral testing of NLP models, extending [CheckList](https://github.com/marcotcr/checklist)
(Ribeiro et al., ACL 2020) with **multi-step transformation chains** and
**grammar-based fuzzing**.

Course research project at Guangdong Technion – Israel Institute of Technology,
advised by Prof. Nazareno Aguirre (Oct 2025 – Feb 2026).

## Idea

Standard CheckList tests apply a *single* perturbation to each input (swap a
name, add a typo, flip a negation) and check the model's response. Real-world
input variation is rarely isolated: typos, entity substitutions, and semantic
modifiers accumulate and interact. This project:

1. **Expands each seed input into a chain** `s0 -> s1 -> ... -> sL`, where each
   step applies a CheckList perturbation operator (typos, punctuation,
   contractions, name/location/number substitution, add/remove negation).
2. **Samples both the perturbation pipeline and the metamorphic property from
   grammars**, using `GrammarFuzzer` from
   [The Fuzzing Book](https://www.fuzzingbook.org/) — no manual test design.
3. **Checks the sampled property** (invariance `INV_FINAL(k)` or directional
   expectations such as `DIR_FINAL(k, EXPECT=MORE_NEGATIVE)`) between the
   original input `s0` and the final transformed input `sL`, through
   CheckList's `INV`/`DIR` test types.

Applied to a HuggingFace sentiment model on the Twitter US Airline Sentiment
dataset, multi-step chains reveal interaction failures and cumulative drift
that single-step perturbations miss.

## Usage

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python metamorphic_testing.py
```

Each run samples a fresh perturbation pipeline and property, builds
transformation chains for 100 seed tweets, runs the CheckList test, and prints
representative passing/failing example pairs.

## Repository layout

| Path | Description |
| --- | --- |
| `metamorphic_testing.py` | Full pipeline: data loading, grammars, chain construction, CheckList integration, qualitative analysis |
| `report/report.pdf` | Project report (method, experiments, comparison with original CheckList, limitations) |
| `report/main.tex` | LaTeX source of the report |

## References

- Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*, ACL 2020
- Zeller et al., *The Fuzzing Book*
- Chen et al., *Metamorphic Testing: A New Approach for Generating Next Test Cases*

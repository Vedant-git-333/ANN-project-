# Student 4 — Intelligence + Dashboard

Two files do everything:

1. **`combine_predictions.py`** — run this first. It reads the other students' output files, puts all 4 models' predictions side by side, works out an "agreement/disagreement" score, a Healthy/Warning/Critical status for each engine, and which sensors matter most. It saves two small CSV files.
2. **`dashboard.py`** — run this second. It's the actual clickable webpage (built with Streamlit) that shows everything from step 1.

## Setup (do this once)

```bash
pip install pandas numpy streamlit plotly
```

## Folder layout it expects

Put the `Student 4` folder inside the repo, next to the others, like this:

```
ANN-project-/
├── Student 1/
│   ├── test_FD001.txt
│   └── train_clean_2D.csv
├── Student 2/
│   └── student2_final_handoff.csv
├── Student 3/
│   └── student3_predictions.csv   <- ask Student 3 for this
└── Student 4/
    ├── combine_predictions.py
    ├── dashboard.py
    └── README.md   (this file)
```

## Run it

```bash
cd "Student 4"
python combine_predictions.py
streamlit run dashboard.py
```

A browser tab opens automatically with the dashboard.

## About Student 3's file

Student 3 has uploaded real data (`Student 3/results/student3_final_handoff.csv`),
so `USE_FAKE_STUDENT3_DATA` is now set to `False` and the script reads their
real LSTM / CNN-LSTM predictions — no more made-up numbers.

One thing to know: LSTM and CNN-LSTM need to look at the *previous 30 cycles*
before they can make a prediction, so they have no output for the first ~30
cycles of each engine's life. That's why the combined file (1,961 rows) has
fewer rows than Student 2's file (2,251 rows) — the merge only keeps rows
where **all 4** models made a prediction. This is expected, not a bug.

## What "disagreement score" means, in plain terms

For every point in time, you have 4 different guesses (ANN, CNN, LSTM, CNN-LSTM)
for how many flights the engine has left. The disagreement score is just how
spread out those 4 guesses are (their standard deviation).

- Low disagreement → the 4 models mostly agree → you can trust the prediction more.
- High disagreement → the models are guessing very differently → treat that
  prediction as uncertain, maybe flag it for a human to double check.

## The project's core question

*"Does disagreement between different neural networks indicate difficult/unreliable
RUL predictions?"*

The script prints a correlation number between disagreement and actual error.
Close to 0 means "not really" — the models can disagree a lot and still both be
roughly right, or agree and both be wrong. Close to 1 means disagreement is a
genuinely useful warning sign.

With the real 4-model data, the result is **Pearson ≈ 0.26, Spearman ≈ 0.36** —
a real, moderate positive relationship. In plain terms: when the models argue
more with each other, the average prediction genuinely does tend to be less
accurate. That's a solid headline finding for your report — use the scatter
plot at the bottom of the dashboard to show it visually.

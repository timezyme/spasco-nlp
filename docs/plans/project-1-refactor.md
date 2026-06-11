# Project 1 Refactor + Repo AI-Reference Scrub

## Context

Project 1 (`1/`, Reuters newswire topic classification, 46 classes) is being polished for employer review of NLP/ML skill. Current state hurts that goal: `part-3.py` is a byte-identical copy of `part-2.py` (the real optimizer study is misnamed `test-3.py`), the classifier/data/plotting code is duplicated ~4x, `part-2.py` crashes under Keras 3 (`ModelCheckpoint('...h5')` requires `.keras`/`.weights.h5`), a 31 MB weights file is committed, evaluation is accuracy-only on an imbalanced 46-class problem, every optimizer config peeks at the test set, and stale summaries describe files that no longer exist. Separately, the user wants **all AI-collaborator references removed repo-wide** (confirmed): a "Collaborators: Google Gemini" line in each docx write-up, AI-dialogue sections in `4/part2/REQUIREMENTS.md`, `claude-fix` filenames in project 5, and Claude Code mentions in two READMEs.

User decisions (confirmed): full refactor + ML upgrades; delete `1/assign1-stephen-pasco.docx` (new `1/README.md` replaces it); strip the Gemini line from all 7 other docx now.

## Phase 0 — Plan file, branch, environment

1. `mkdir -p docs/plans` and copy this plan to `docs/plans/project-1-refactor.md` (user-requested location; plan mode blocked writing it there directly).
2. Branch: `git checkout -b project-1-refactor` (git status is clean).
3. Create venv with **python3.12**, not the 3.14 default (TensorFlow has no cp314 wheels): `/opt/homebrew/bin/python3.12 -m venv venv`, then pip install `nltk spacy scikit-learn tensorflow keras pandas matplotlib seaborn jupyter python-docx`. Update the `CLAUDE.md` env section to say python3.12 (file is gitignored, local-only).
4. Gate: `./venv/bin/python -c "import keras, sklearn; ..."` shows Keras 3.x; `reuters.load_data(num_words=10000)` returns 8982/2246 (downloads ~2 MB, needs network once).

## Phase 1 — Project 1 hygiene (no new code)

- `git rm 1/part-3/part-3.py` (exact duplicate), `git rm 1/part-2/best_model_weights.h5`, `git rm 1/assign1-stephen-pasco.docx`.
- Add `*.h5`, `*.weights.h5`, `*.keras` to root `.gitignore`.
- Create `1/archived/`; `git mv` into it: old `part-1/part-1.py`, `part-2/part-2.py`, `part-3/test-3.py` (after Phase 3 replaces them), `SUMMARY1a.md`, `IMPROVEMENTS.md`, `model_optimization_summary.txt`, `task_summary.md`, and the notebook renamed `assign1-question1.ipynb` → `chollet-3.5-reference.ipynb` (it is verbatim Chollet ch. 3.5 teaching code on Keras-2 APIs; do not modernize third-party material).
- Delete regenerable outputs: both `part1b_results.txt`, `reuters_training_history.png`, `results-plot.png`, `optimizer_results.txt`, old `optimizer_comparison.png`.

## Phase 2 — Shared module `1/reuters_common.py` (~300 lines, functions only)

Top of module: `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` before keras import; `matplotlib.use("Agg")` before pyplot; constants `NUM_WORDS=10000`, `NUM_CLASSES=46`, `VAL_SIZE=1000`, `SEED=42`.

- `ReutersData` frozen dataclass: multi-hot float32 x splits, one-hot y splits, integer-label y splits.
- `vectorize_sequences(sequences, dimension)` — multi-hot encoding.
- `load_reuters(...)` — Chollet protocol: first 1000 train samples as validation.
- `build_baseline_model(...)` — `keras.Input` → Dense 64 relu ×2 → softmax; rmsprop.
- `build_improved_model(..., optimizer=None)` — 256/128/64 + BatchNorm + dropout 0.4/0.3/0.2; default Adam(1e-3); `optimizer` param reused by part 3.
- `train_model(...)` — fit with `validation_data`, `verbose=0` enforced here once.
- `best_epoch_from_history(history)` — 1-indexed argmax of `val_accuracy`.
- `plot_training_history(...)` — dual-panel acc/loss with optimal-epoch marker; savefig+close, never show.
- `plot_confusion_matrix(...)` — row-normalized 46×46 imshow, no per-cell annotations, matplotlib only.
- `predict_classes`, `metrics_block` (accuracy, macro-F1, weighted-F1, `classification_report(zero_division=0)`), `majority_class_baseline` (~36%), `shuffled_labels_baseline` (~19%, seeded `np.random.default_rng(42)`), `tfidf_logreg_baseline` (sparse counts → TfidfTransformer → `LogisticRegression(max_iter=1000, random_state=42)`), `versions_header()` for output files.

Part scripts import it via `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` (hyphenated dirs can't be packages; no packaging). All output paths derive from `Path(__file__).resolve().parent` — CWD-relative paths caused the original scattered results files.

Keras 3 fixes baked in: no `input_shape=` kwarg (use `keras.Input`), no `ModelCheckpoint` at all (`EarlyStopping(restore_best_weights=True)` covers it), no `acc`/`val_acc` dual-key fallbacks, no bare `except`, no emoji in output, `keras.utils.set_random_seed(42)` in every script.

## Phase 3 — Rewrite the three part scripts

- **`1/part-1/part-1.py`** (~130 lines): baseline 64-64 model, 20 epochs batch 512 → plot → optimal epoch by max val accuracy → retrain from scratch at that count → single test evaluation → accuracy, macro-F1, shuffled + majority baselines → `part-1_output.txt` + `training_history.png`. Gate: runs ~1–2 min, test acc 0.77–0.80.
- **`1/part-2/part-2.py`** (~180 lines): improved model with EarlyStopping(`val_accuracy`, patience 5, restore_best_weights) + ReduceLROnPlateau; re-train the 64-64 baseline in-run; run TF-IDF+LogReg classical baseline; output a 3-row comparison table (val acc / test acc / macro-F1 / train seconds) + per-class report + `confusion_matrix.png`. Gate: ~2–4 min, no model artifact files appear, improved ≈ 0.79–0.81, TF-IDF+LogReg ≈ 0.79–0.81 in seconds (macro-F1 will honestly sit ~0.55–0.70 — imbalanced classes; README frames this).
- **`1/part-3/part-3.py`** (~210 lines, replaces `test-3.py`): `CONFIGS` list at top — 10 configs, 2 per family (RMSprop, Adam, Adamax, AdamW, Nadam; keep one lr=1e-4 config for the "learning rate dominates optimizer family" finding) — plus `RUN_ONLY: str | None = None` filter to re-run a single config when iterating. **Protocol fix: rank on validation accuracy only; evaluate exactly one winner on the test set.** Plain file writes to `part-3_output.txt` (f-string columns, no pandas, no TeeOutput stdout hack); 3-panel `optimizer_comparison.png` (val-acc curves best-per-family, ranked bar chart, Adam LR comparison). Docstring documents the sweep as the deliberate exception to the one-config-per-run convention (the comparison table *is* this part's deliverable; comparison plots need per-epoch histories from all configs). Gate: ~5–12 min, exactly one test number in the output file.

## Phase 4 — Documentation

- **New `1/README.md`** (employer-facing): problem + dataset; approach per part; consolidated results table (baseline MLP / improved MLP / TF-IDF+LogReg / optimizer winner) using **only freshly generated numbers**; plot links; key findings (classical ≈ deep at this representation, multi-hot ceiling ~80–82%, lr matters more than optimizer family, macro-F1 vs accuracy under imbalance); evaluation-protocol note (select on val, single test eval); how to run; credit Chollet ch. 3.5 as the starting point, pointing at `archived/chollet-3.5-reference.ipynb`.
- **Root `README.md`**: update the Project 1 section (currently points only at the notebook) to point at `1/README.md` and the three parts.
- Wording rules: never use assignment/course/professor/TA/student; no emojis; no AI-tool mentions.

## Phase 5 — Repo-wide AI-collaborator scrub (confirmed scope)

- **Root `README.md`**: remove the `CLAUDE.md — AI assistant configuration` tree line and the "Claude Code integration" mention.
- **`2/README.md`**: remove/rewrite the three CLAUDE.md-referencing lines.
- **`4/part2/REQUIREMENTS.md`**: delete the ` ```gemini ` and ` ```claude-sonnet ` dialogue sections (~lines 163–383) and de-attribute any remaining mentions; keep actual requirements content.
- **`4/part2/ALL-PARTS.ipynb`**: delete the `COLLABORATORS: GOOGLE GEMINI FLASH...` line from markdown cell 0 (edit JSON via python, preserve outputs).
- **Project 5 renames**: `5/problem3/problem3-claude-fix.py` → `problem3-fix.py`, `...claude-fix_output.txt` → `problem3-fix_output.txt`; update self-references inside the script (usage comment, output path) and the references in `5/PROBLEM3_SOLUTION_MEMORY.md`; update the `*-claude-fix.py` naming convention in local `CLAUDE.md` to `*-fix.py`.
- **Docx strip (projects 2, 4, 5, 6, 7, 8, 9)**: with `./venv/bin/python` + python-docx, remove the paragraph containing "Collaborators" from each; verify by re-extracting text that no Collaborators/Gemini hit remains and the rest of the document is intact.
- Final gates: `git grep -iE 'claude|anthropic|gemini|chatgpt|openai|copilot|collaborator'` over tracked files returns nothing; docx re-extraction loop returns "none" for all; `grep -inE '\bassignment|course|professor|\bTA\b|student'` over all new/changed text files returns nothing. (Pre-scrub content survives in git history — accepted, consistent with the June 2026 course scrub.)

## Commits and verification

Small conventional commits on `project-1-refactor`, roughly: plan copy → hygiene → common module → part-1 → part-2 → part-3 → project 1 docs → AI scrub (text) → AI scrub (docx + renames). **No AI co-author trailers on any commit.** Merge to main only when the user approves.

End-to-end verification: run all three part scripts in order with `./venv/bin/python` from repo root; outputs land beside their scripts; numbers in sanity ranges above; `1/README.md` numbers match the fresh `*_output.txt` files; all README links resolve; `git status` clean; total re-run ~10–18 min on this machine.

## Out of scope (deliberately)

No god-class port, no argparse/YAML config, no MLflow/W&B/pytest scaffolding, no packaging, no embeddings/LSTM/BERT implementations (README mentions them as next steps — that is the judgment signal), no plot theming, no modernizing the Chollet notebook, no git-history rewrite for the 31 MB blob (note `git filter-repo` as optional follow-up only).

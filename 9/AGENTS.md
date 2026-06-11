# Repository Guidelines

## Project Structure & Module Organization
- Primary work happens in the notebooks: `Assign9.ipynb` for the project deliverable and `Assign9_test.ipynb` for quick experiments.
- Data lives in `data/news_data.csv`; keep derived artifacts (plots, model weights) in a temporary folder or ignore them via `.gitignore`.
- Supporting references: `problem-requirements.txt` outlines tasks; `arxiv_2510.01115_sections-consolidated.json` is a readable source document; the `.docx` file contains the write-up text.

## Build, Test, and Development Commands
- Create/activate an environment (Python 3.10+ recommended):
  - `python -m venv venv`
  - `source venv/bin/activate`
- Install common dependencies used across the tasks (adjust as needed): `pip install nltk spacy scikit-learn tensorflow pandas matplotlib jupyter`.
- Download the SpaCy model for NER: `python -m spacy download en_core_web_sm`.
- Run notebooks interactively: `jupyter lab` (or `jupyter notebook`).
- Execute a notebook non-interactively to verify it runs end-to-end: `jupyter nbconvert --to notebook --execute Assign9.ipynb --output /tmp/assign9_run.ipynb`.

## Coding Style & Naming Conventions
- Follow PEP 8 for Python: 4-space indents, snake_case for functions/variables, CapWords for classes.
- Keep notebook cells deterministic: set seeds (`random_state=42`) and avoid downloading inside hot paths so reruns are reproducible.
- Prefer small, well-named helper functions in code cells instead of sprawling inline logic; keep regex patterns and model configs grouped near their usage with brief comments.

## Testing Guidelines
- For quick checks, add lightweight assertions/print summaries in `Assign9_test.ipynb` before moving code into `Assign9.ipynb`.
- When adding scripts/modules, mirror notebook logic in functions and cover with `pytest` or inline checks (e.g., verifying entity extraction counts, train/test splits, and accuracy thresholds).
- Record key metrics (e.g., NER precision notes, classification accuracy) in the notebook output so reviewers can confirm results without rerunning heavy steps.

## Commit & Pull Request Guidelines
- Commits: concise, present-tense summaries (e.g., “Add spaCy NER baseline”, “Tune TF-IDF features”); group related notebook/script changes together.
- Pull requests: include a short description of the goal, datasets touched, commands run, and observed metrics (accuracy plots, final test accuracy). If notebooks were executed, mention the environment (Python version, model downloads) and any non-default parameters.
- Avoid committing large intermediate files (model weights, `.ipynb_checkpoints`, `/tmp` outputs); keep the repo lean.

## Security & Configuration Tips
- Keep API keys or credentials out of notebooks; prefer environment variables if needed.
- The repository is self-contained; network access is only required for initial model downloads (e.g., SpaCy weights). Cache models inside `venv` or a local model cache rather than committing them.

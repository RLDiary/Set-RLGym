# Repository Guidelines

## Project Structure & Modules
- `Set/agent.py`: LLM gameplay loop (1-based indices in prompts, converts internally).
- `Set/env.py`: Core game rules, board state, Cairo rendering config.
- `Set/Utils/generate_board.py`: Cairo drawing utilities for cards/boards.
- `Set/templates/`: Jinja2 prompt templates used by the agent.
- `Assets/`: Example artifacts and docs; `Specs/`: technical notes/specifications.
- Add tests under `tests/` (see Testing Guidelines).

## Build, Run, and Development
- Python 3.12 required (see `pyproject.toml`).
- Install deps (preferred): `uv sync`
- Alt install (PEP 621): `pip install -e .`
- Run agent demo: `python run.py --backend openai --verbose`
- Run env demo (renders PNGs): `python Set/env.py`
- Environment: set `OPENAI_API_KEY` (and optionally `OPENROUTER_API_KEY`). `.env` is supported via `python-dotenv`.

## Coding Style & Conventions
- Follow PEP 8; 4-space indentation; keep lines ≤ 100 chars.
- Use type hints and module docstrings (see existing files for patterns).
- Naming: modules `snake_case.py`, classes `CapWords`, functions/vars `snake_case`.
- Imports grouped stdlib/third-party/local with clear separation.
- Keep 1-based visual indexing for UI/LLM, 0-based internal indices in `env.py`.

## Testing Guidelines
- Framework: `pytest` (add as a dev dependency if missing).
- Layout: `tests/test_env.py`, `tests/test_agent.py`, etc.; name tests `test_*`.
- Cover core logic: `is_set`, `all_sets`, `SetEnv.select/deal_three`, indexing conversions.
- Run tests: `pytest -q`; for deterministic cases, pass a fixed `seed`.

## Commit & PR Guidelines
- Commits: short, imperative subject lines (e.g., “Add env rendering config”).
- Scope one logical change per commit; include brief body if behavior changes.
- PRs: clear description, linked issues, reproduction steps, and before/after notes.
- For rendering/agent changes: attach sample images (`--save-prefix`) and seed used.
- Update README and templates if behavior or prompts change.

## Security & Configuration
- Never commit API keys. Use `OPENAI_API_KEY` via environment or `.env` (gitignored).
- Network calls are optional during local development; mock in tests where possible.

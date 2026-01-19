# Repository Guidelines

## Project Structure & Module Organization
- `ludicrous.py` contains the TinyJit Mandelbrot benchmark and demo entrypoint.
- `main.py` is a minimal hello-world entrypoint (handy for quick environment checks).
- `pyproject.toml` defines the project metadata and Python dependencies.
- `uv.lock` pins dependencies for repeatable installs when using `uv`.

## Build, Test, and Development Commands
- `python main.py` runs the lightweight sanity check.
- `python ludicrous.py` runs the Mandelbrot benchmark with TinyGrad/TinyJit.
- `uv run python ludicrous.py` does the same using the locked environment (preferred if you use `uv`).
- `uv sync` installs dependencies from `pyproject.toml`/`uv.lock`.

## Coding Style & Naming Conventions
- Python 3.12; follow PEP 8 with 4-space indentation.
- Use snake_case for functions/variables (e.g., `mandelbrot_tinygrad`).
- Keep scripts runnable via `if __name__ == "__main__":` blocks.
- Prefer short, descriptive names for benchmark labels and constants (e.g., `W`, `H`, `ITERS`).

## Testing Guidelines
- No formal test suite yet. If you add tests, place them under `tests/` and name files `test_*.py`.
- Favor fast, deterministic checks for benchmark outputs or shape expectations.

## Commit & Pull Request Guidelines
- No commit history exists yet; use concise, imperative commit messages (e.g., `Add TinyJit benchmark runner`).
- PRs should include: a short summary, the commands run, and any performance numbers when relevant.
- If behavior changes, include before/after timings or device notes (e.g., `CPU`, `METAL`, `CUDA`).

## Configuration Tips
- You can select backends via env vars like `CUDA=1`, `METAL=1`, or `CL=1` when running `ludicrous.py`.
- If benchmarking on GPU, keep runs consistent and note the device in results.

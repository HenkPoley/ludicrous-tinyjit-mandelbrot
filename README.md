# Ludicrous Python Speeds (TinyJit Demo)

This repo compares a pure-Python Mandelbrot benchmark with tinygrad and TinyJit.

## Quick Start (uv)
If you are starting fresh:

```bash
uv init
uv add tinygrad
uv run python ludicrous.py
```

Using the existing project setup:

```bash
uv sync
uv run python ludicrous.py
```

## Run Options
- Default run: `uv run python ludicrous.py`
- Full workload: `uv run python ludicrous.py --full`
- Skip Python baseline: `uv run python ludicrous.py --skip-python`
- Pick device: `uv run python ludicrous.py --device METAL` (also works with `CPU`, `CL`, `CUDA`, etc.)

## Notes
- `main.py` is a simple sanity check: `python main.py`
- TinyJit warms up before timing so that the replay runs show the speedup.

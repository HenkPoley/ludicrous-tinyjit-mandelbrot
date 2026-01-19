# Ludicrous Python Speeds (TinyJit Demo)

This repo compares a pure-Python Mandelbrot benchmark with tinygrad and TinyJit.

## Quick Start (uv)
If you are starting fresh:

```bash
uv init
uv add tinygrad
uv run python ludicrous-mandelbrot.py
```

Using the existing project setup:

```bash
uv sync
uv run python ludicrous-mandelbrot.py
```

## Run Options
- Default run: `uv run python ludicrous-mandelbrot.py`
- Full workload: `uv run python ludicrous-mandelbrot.py --full`
- Skip Python baseline: `uv run python ludicrous-mandelbrot.py --skip-python`
- Pick device: `uv run python ludicrous-mandelbrot.py --device METAL` (also works with `CPU`, `CL`, `CUDA`, etc.)
- Run all available devices: `uv run python ludicrous-mandelbrot.py --device all`
- Control runs: `uv run python ludicrous-mandelbrot.py --runs-tiny 1 --runs-jit 5`
- Control warmup size: `uv run python ludicrous-mandelbrot.py --warmup-size same` (options: `same`, `small`)
- Chunk JIT capture: `uv run python ludicrous-mandelbrot.py --jit-chunk 25`
- Verify outputs: `uv run python ludicrous-mandelbrot.py --verify`

## Same-Kernel Example
This variant uses the exact same kernel function for Python scalars and tinygrad Tensors.

```bash
uv run python ludicrous-mandelbrot-samecode.py
uv run python ludicrous-mandelbrot-samecode.py --skip-python
```

## Notes
- TinyJit warms up before timing so that the replay runs show the speedup (warmup timings are printed).

import argparse
import time
from tinygrad import Tensor, TinyJit, Device

def mandelbrot_python(width=640, height=360, iters=100):
  """Baseline: pure Python (intentionally slow)"""
  out = [[0]*width for _ in range(height)]
  for y in range(height):
    cy = (y / height) * 2.0 - 1.0
    for x in range(width):
      cx = (x / width) * 3.5 - 2.5
      zr = 0.0
      zi = 0.0
      n = 0
      while n < iters and (zr*zr + zi*zi) <= 4.0:
        zr, zi = (zr*zr - zi*zi + cx), (2*zr*zi + cy)
        n += 1
      out[y][x] = n
  return out

def mandelbrot_step(zr, zi, cr, ci, alive, counts):
  zr2 = zr*zr
  zi2 = zi*zi
  mag2 = zr2 + zi2
  still = (mag2 <= 4.0) * alive  # 1.0 where still active, else 0.0

  # Increment counts where still active
  counts = counts + still

  # Update z where still active, else keep previous z
  new_zr = zr2 - zi2 + cr
  new_zi = (zr*zi*2.0) + ci
  zr = still.where(new_zr, zr)
  zi = still.where(new_zi, zi)

  alive = still
  return zr, zi, alive, counts


def mandelbrot_tinygrad(width=640, height=360, iters=100, device=None, realize_each=False):
  """
  Tensorised Mandelbrot. All ops are lazy until realized.
  (So we force realization at the end for benchmarking.)
  """
  device = device or Device.DEFAULT

  # Build pixel grid in [-2.5, 1.0] x [-1.0, 1.0]
  xs = (Tensor.arange(width, device=device).float() / width) * 3.5 - 2.5
  ys = (Tensor.arange(height, device=device).float() / height) * 2.0 - 1.0
  cr = xs.reshape(1, width).expand(height, width)
  ci = ys.reshape(height, 1).expand(height, width)

  zr = Tensor.zeros(height, width, device=device)
  zi = Tensor.zeros(height, width, device=device)
  alive = Tensor.ones(height, width, device=device)  # 1.0 where still iterating
  counts = Tensor.zeros(height, width, device=device)

  for _ in range(iters):
    zr, zi, alive, counts = mandelbrot_step(zr, zi, cr, ci, alive, counts)
    if realize_each:
      # Keep the graph small for eager-style runs.
      zr = zr.realize()
      zi = zi.realize()
      counts = counts.realize()
      alive = alive.realize()

  # Force execution so timing is real (tinygrad is lazy)
  # (Calling .numpy() also forces realization.)
  return counts.realize()

# TinyJit wrapper: after capture, subsequent calls replay without Python overhead
# (The tinygrad docs show this is where the big speedup comes from.)
@TinyJit
def mandelbrot_tinyjit(width, height, iters, device):
  return mandelbrot_tinygrad(width, height, iters, device)


@TinyJit
def mandelbrot_tinyjit_chunk(zr, zi, cr, ci, alive, counts, iters):
  for _ in range(iters):
    zr, zi, alive, counts = mandelbrot_step(zr, zi, cr, ci, alive, counts)
  return zr, zi, alive, counts


def mandelbrot_tinyjit_chunked(width, height, iters, device, chunk):
  device = device or Device.DEFAULT
  xs = (Tensor.arange(width, device=device).float() / width) * 3.5 - 2.5
  ys = (Tensor.arange(height, device=device).float() / height) * 2.0 - 1.0
  cr = xs.reshape(1, width).expand(height, width)
  ci = ys.reshape(height, 1).expand(height, width)

  cr = cr.contiguous()
  ci = ci.contiguous()
  zr = Tensor.zeros(height, width, device=device).contiguous()
  zi = Tensor.zeros(height, width, device=device).contiguous()
  alive = Tensor.ones(height, width, device=device).contiguous()
  counts = Tensor.zeros(height, width, device=device).contiguous()

  if chunk < iters:
    remaining = iters
    while remaining > 0:
      step = min(chunk, remaining)
      zr, zi, alive, counts = mandelbrot_tinyjit_chunk(zr, zi, cr, ci, alive, counts, step)
      remaining -= step
  else:
    zr, zi, alive, counts = mandelbrot_tinyjit_chunk(zr, zi, cr, ci, alive, counts, iters)

  return counts.realize()

def time_run(label, fn, runs=3, device=None):
  ts = []
  for _ in range(runs):
    t0 = time.perf_counter()
    _ = fn()
    # If you're on a GPU backend, synchronizing makes timings more honest.
    # (Runtimes expose a synchronize method.)
    Device[device or Device.DEFAULT].synchronize()
    ts.append(time.perf_counter() - t0)
  print(f"{label}: " + ", ".join(f"{t*1000:.1f} ms" for t in ts))


def summarize_output(data):
  if isinstance(data, Tensor):
    h, w = data.shape
    total = float(data.sum().item())
    return total, float(data[0, 0].item()), float(data[h - 1, w - 1].item())
  h = len(data)
  w = len(data[0]) if h else 0
  total = float(sum(sum(row) for row in data))
  return total, float(data[0][0]), float(data[h - 1][w - 1])


def verify_outputs(py_out, tg_out, tj_out):
  py_stats = summarize_output(py_out)
  tg_stats = summarize_output(tg_out)
  tj_stats = summarize_output(tj_out)
  ok = py_stats == tg_stats == tj_stats
  print(f"Verify outputs: {'OK' if ok else 'MISMATCH'}")
  if not ok:
    print(f"  python: {py_stats}")
    print(f"  tinygrad: {tg_stats}")
    print(f"  tinyjit: {tj_stats}")
  return ok

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="TinyJit Mandelbrot benchmark.")
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=180)
  parser.add_argument("--iters", type=int, default=80)
  parser.add_argument("--device", type=str, default=Device.DEFAULT)
  parser.add_argument("--full", action="store_true", help="Use the larger 640x360x150 defaults.")
  parser.add_argument("--runs-python", type=int, default=1)
  parser.add_argument("--runs-tiny", type=int, default=3)
  parser.add_argument("--runs-jit", type=int, default=5)
  parser.add_argument("--jit-chunk", type=int, default=0, help="Chunk JIT iterations to avoid huge captures.")
  parser.add_argument("--warmup-size", choices=["small", "same"], default=None)
  parser.add_argument("--verify", action="store_true", help="Compare outputs across Python, tinygrad, and TinyJit.")
  parser.add_argument("--skip-python", action="store_true")
  parser.add_argument("--skip-tiny", action="store_true")
  parser.add_argument("--skip-jit", action="store_true")
  args = parser.parse_args()

  if args.full:
    args.width, args.height, args.iters = 640, 360, 150
    if args.jit_chunk == 0:
      args.jit_chunk = 25
  if args.warmup_size is None:
    args.warmup_size = "small" if args.full else "same"

  print("Default device:", Device.DEFAULT)  # docs: CUDA on GPU boxes, else CPU
  print("Available devices:", list(Device.get_available_devices()))  # example output in issue

  # Pick one of: "CPU", "METAL", "CUDA", "CL", "WEBGPU", etc.
  # You can also enable/force backends with env vars like CUDA=1, METAL=1, CL=1, CPU=1.
  devices = list(Device.get_available_devices()) if args.device == "all" else [args.device]
  if not devices:
    devices = [Device.DEFAULT]

  W, H, ITERS = args.width, args.height, args.iters

  for device in devices:
    print("Selected device:", device)

    if args.verify:
      if args.skip_python or args.skip_tiny or args.skip_jit:
        print("Verify outputs: skipped (requires Python, tinygrad, and TinyJit).")
      else:
        print("Verifying outputs ...")
        py_out = mandelbrot_python(W, H, ITERS)
        tg_out = mandelbrot_tinygrad(W, H, ITERS, device)
        tj_out = (
          mandelbrot_tinyjit_chunked(W, H, ITERS, device, chunk=args.jit_chunk)
          if args.jit_chunk
          else mandelbrot_tinyjit(W, H, ITERS, device)
        )
        Device[device].synchronize()
        verify_outputs(py_out, tg_out, tj_out)

    # 1) Pure Python baseline (slow)
    if not args.skip_python:
      print("Running pure Python baseline ...")
      time_run("Pure Python (double loop)", lambda: mandelbrot_python(W, H, ITERS), runs=args.runs_python, device=device)

    # 2) tinygrad without JIT (still benefits from kernel fusion / backend)
    if not args.skip_tiny:
      print("Running tinygrad (eager-ish) ...")
      time_run(
        "tinygrad (eager-ish, realized)",
        lambda: mandelbrot_tinygrad(W, H, ITERS, device, realize_each=True),
        runs=args.runs_tiny,
        device=device,
      )

    # 3) tinygrad + TinyJit
    # Expectation (per docs): first runs include capture/compile, then it gets *much* faster.
    if not args.skip_jit:
      # Warm up small workload to trigger kernel compile/capture before the JIT run.
      if args.warmup_size == "same":
        w_warm, h_warm, i_warm = W, H, ITERS
      else:
        w_warm, h_warm, i_warm = 64, 48, 20
      print(f"Warmup ({w_warm}x{h_warm}x{i_warm}) ...")
      time_run(
        "Warmup tinygrad",
        lambda: mandelbrot_tinygrad(w_warm, h_warm, i_warm, device),
        runs=1,
        device=device,
      )
      time_run(
        "Warmup TinyJit",
        lambda: (
          mandelbrot_tinyjit_chunked(w_warm, h_warm, i_warm, device, chunk=args.jit_chunk)
          if args.jit_chunk
          else mandelbrot_tinyjit(w_warm, h_warm, i_warm, device)
        ),
        runs=1,
        device=device,
      )
      print("Running tinygrad + TinyJit ...")
      if args.jit_chunk:
        print(f"TinyJit chunk size: {args.jit_chunk}")
      time_run(
        "tinygrad + TinyJit",
        lambda: (
          mandelbrot_tinyjit_chunked(W, H, ITERS, device, chunk=args.jit_chunk)
          if args.jit_chunk
          else mandelbrot_tinyjit(W, H, ITERS, device)
        ),
        runs=args.runs_jit,
        device=device,
      )

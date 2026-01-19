import argparse
import time

from tinygrad import Device, TinyJit
from tinygrad.tensor import Tensor


def to_mask(x):
  try:
    return x.float()
  except AttributeError:
    return 1.0 if x else 0.0


def mandelbrot_kernel(cr, ci, iters):
  """
  Same kernel for Python scalars and tinygrad Tensors.
  It works with floats (single point) or Tensor grids (vectorized).
  """
  zr = cr * 0.0
  zi = ci * 0.0
  counts = cr * 0.0
  alive = cr * 0.0 + 1.0

  for _ in range(iters):
    zr2 = zr * zr
    zi2 = zi * zi
    mag2 = zr2 + zi2
    mask = to_mask(mag2 <= 4.0) * alive
    counts = counts + mask

    new_zr = zr2 - zi2 + cr
    new_zi = (zr * zi * 2.0) + ci
    zr = mask * new_zr + (1.0 - mask) * zr
    zi = mask * new_zi + (1.0 - mask) * zi
    alive = mask

  return counts


def build_grid(width, height, device):
  xs = (Tensor.arange(width, device=device).float() / width) * 3.5 - 2.5
  ys = (Tensor.arange(height, device=device).float() / height) * 2.0 - 1.0
  cr = xs.reshape(1, width).expand(height, width)
  ci = ys.reshape(height, 1).expand(height, width)
  return cr.contiguous(), ci.contiguous()


def mandelbrot_python(width=320, height=180, iters=80):
  out = [[0] * width for _ in range(height)]
  for y in range(height):
    cy = (y / height) * 2.0 - 1.0
    for x in range(width):
      cx = (x / width) * 3.5 - 2.5
      out[y][x] = int(mandelbrot_kernel(cx, cy, iters))
  return out


def mandelbrot_tinygrad(width=320, height=180, iters=80, device=None):
  device = device or Device.DEFAULT
  cr, ci = build_grid(width, height, device)
  return mandelbrot_kernel(cr, ci, iters).realize()


@TinyJit
def mandelbrot_tinyjit(width, height, iters, device):
  device = device or Device.DEFAULT
  cr, ci = build_grid(width, height, device)
  return mandelbrot_kernel(cr, ci, iters)


def time_run(label, fn, runs=3, device=None):
  ts = []
  for _ in range(runs):
    t0 = time.perf_counter()
    _ = fn()
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
  parser = argparse.ArgumentParser(description="Same-kernel Mandelbrot benchmark.")
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=180)
  parser.add_argument("--iters", type=int, default=80)
  parser.add_argument("--device", type=str, default=Device.DEFAULT)
  parser.add_argument("--runs-python", type=int, default=1)
  parser.add_argument("--runs-tiny", type=int, default=3)
  parser.add_argument("--runs-jit", type=int, default=5)
  parser.add_argument("--verify", action="store_true", help="Compare outputs across Python, tinygrad, and TinyJit.")
  parser.add_argument("--skip-python", action="store_true")
  parser.add_argument("--skip-tiny", action="store_true")
  parser.add_argument("--skip-jit", action="store_true")
  args = parser.parse_args()

  devices = list(Device.get_available_devices()) if args.device == "all" else [args.device]
  if not devices:
    devices = [Device.DEFAULT]

  for device in devices:
    print("Selected device:", device)

    if args.verify:
      if args.skip_python or args.skip_tiny or args.skip_jit:
        print("Verify outputs: skipped (requires Python, tinygrad, and TinyJit).")
      else:
        print("Verifying outputs ...")
        py_out = mandelbrot_python(args.width, args.height, args.iters)
        tg_out = mandelbrot_tinygrad(args.width, args.height, args.iters, device)
        tj_out = mandelbrot_tinyjit(args.width, args.height, args.iters, device).realize()
        Device[device].synchronize()
        verify_outputs(py_out, tg_out, tj_out)

    if not args.skip_python:
      time_run(
        "Pure Python (same kernel)",
        lambda: mandelbrot_python(args.width, args.height, args.iters),
        runs=args.runs_python,
        device=device,
      )

    if not args.skip_tiny:
      time_run(
        "tinygrad (same kernel)",
        lambda: mandelbrot_tinygrad(args.width, args.height, args.iters, device),
        runs=args.runs_tiny,
        device=device,
      )

    if not args.skip_jit:
      time_run(
        "tinygrad + TinyJit (same kernel)",
        lambda: mandelbrot_tinyjit(args.width, args.height, args.iters, device).realize(),
        runs=args.runs_jit,
        device=device,
      )

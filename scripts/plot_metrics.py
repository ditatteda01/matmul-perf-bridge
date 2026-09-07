import sys
import matplotlib.pyplot as plt
from pathlib import Path
from parser import parse_metrics_log
from parser import METHODS

def flops(m, n, k):
    # 2*p*q*r floating point ops (multipy + add) for a naive matmul
    return 2 * m * n * k

def plot_total_time(records, dims, save_path="benchmark_total_time.png"):
    x = [m for m, n, k in dims]

    methods = {}
    for name, key in METHODS:
        if name.startswith("CPP"):
            methods[name] = [
                r[key]["total_s"] if r[key] else r[key]
                for r in records
            ]
        else:
            methods[name] = [r[key] for r in records]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y in methods.items():
        xs = [xi for xi, yi in zip(x, y) if yi is not None]
        ys = [yi for yi in y if yi is not None]
        if ys:
            ax.plot(xs, ys, marker="o", label=label)
    
    ax.set_yscale("log")
    ax.set_xlabel("Problem size M=N=K")
    ax.set_ylabel("Total wall-clock time (s)")
    ax.set_title("Matmul benchmark: total time vs. problem size")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

def plot_cpp_alg_time(records, dims, save_path="benchmark_cpp_alg_time.png"):
    x = [m for m, n, k in dims]

    methods = {}
    for name, key in METHODS:
        if name.startswith("CPP"):
            methods[name] = [
                r[key]["alg_ms"] if r[key] else r[key]
                for r in records
            ]

    
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y in methods.items():
        xs = [xi for xi, yi in zip(x, y) if yi is not None]
        ys = [yi for yi in y if yi is not None]
        if ys:
            ax.plot(xs, ys, marker="o", label=label)
    
    ax.set_yscale("log")
    ax.set_xlabel("Problem size M=N=K")
    ax.set_ylabel("Algorithm time (ms)")
    ax.set_title("Matmul benchmark: C++ Algorithm time vs. problem size")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

def plot_cpp_GFLOPs(records, dims, save_path="benchmark_cpp_gflops.png"):
    gflop = [flops(*dim) * 1e-9 for dim in dims]
    x = [m for m, n, k in dims]

    methods = {}
    for name, key in METHODS:
        if name.startswith("CPP"):
            methods[name] = [
                r[key]["alg_ms"] if r[key] else r[key]
                for r in records
            ]

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y in methods.items():
        xs = [xi for xi, yi in zip(x, y) if yi is not None]
        ys = [gfp / yi * 1e3 for gfp, yi in zip(gflop, y) if yi is not None]
        if ys:
            ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("Problem size M=N=K")
    ax.set_ylabel("GFLOPs")
    ax.set_title("Matmul benchmark: C++ GFLOPs vs. problem size")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

if __name__ == "__main__":
    try:
        log_path = sys.argv[1] if len(sys.argv) > 1 else Path("log","metrics.log")
        out_dir = sys.argv[2] if len(sys.argv) > 2 else Path("png")

        _, dims, records = parse_metrics_log(log_path)
        p1 = plot_total_time(records, dims, save_path=out_dir/"benchmark_total_time.png")
        p2 = plot_cpp_alg_time(records, dims, save_path=out_dir/"benchmark_cpp_alg_time.png")
        p3 = plot_cpp_GFLOPs(records, dims, save_path=out_dir/"benchmark_cpp_GFLOPs.png")
        print(f"Saved: {p1} | {p2} | {p3}")

    except Exception as e:
        print(e, file=sys.stderr)
import sys
import re
from pathlib import Path
from parser import parse_metrics_log
from parser import METHODS


def update_benchmark_table_basic(parsed_data, md_path="benchmark_results.md"):
    """Update the benchmark C++ basic approach table.
    
    Args:
        md_path: Target file to be updated.
        parsed_data: Tuple of lists (epochs, dims, records) prepared by *parse_metrics_log*.
    """
    try:
        md_table = "| Epoch | Dimensions (p, q, r) |"
        for m, _ in METHODS:
            if m.startswith("CPP"):
                md_table += " " + m + " Total(s) / Alg(s) |"
            else:
                md_table += " " + m + " Total(s) |"
        md_table += "\n"

        cols = len(md_table.strip(" |\n").split("|"))
        md_table += "| " + " | ".join(["---" for _ in range(cols)]) + " |\n"

        epochs, dims, records = parsed_data
        for e, d, r in zip(epochs, dims, records):
            md_table += f"| {e} | {d[0]}x{d[1]}x{d[2]} |"
            for method, key in METHODS:
                if method.startswith("CPP"):
                    md_table += f" {r[key]['total_s']:.6f} / {r[key]['alg_ms']/1000:.6f} |"
                else:
                    s = f" {r[key]:.6f} |" if r[key] else " skipped |"
                    md_table += s
            md_table += "\n"

        with open(md_path, "r", encoding="utf-8") as fr:
            md_text = fr.read()

        pattern = r"(##\s*Baseline\s*C\+\+\s*Implementations\n(?:(?!\|).*\n)*)(?:\|.*\n)*"
        updated_md_text = re.sub(pattern, lambda m: f"{m.group(1)}{md_table}", md_text)

        with open(md_path, "w", encoding="utf-8") as fw:
            fw.write(updated_md_text)

        return f"Successfully updated table in {md_path}"

    except Exception as e:
        return f"Error in update table: {e}"


if __name__ == "__main__":
    try:
        log_path = sys.argv[1] if len(sys.argv) > 1 else Path("docs", "metrics.log")
        md_path = sys.argv[2] if len(sys.argv) > 2 else Path("docs", "benchmark_results.md")

        print(update_benchmark_table_basic(parse_metrics_log(log_path), md_path))
    
    except Exception as e:
        print(e, file=sys.stderr)
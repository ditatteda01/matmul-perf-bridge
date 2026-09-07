import re
import sys
from pathlib import Path

METHODS = [
    ("Numpy matmul", "py_numpy"),
    ("Python for-loop", "py_loop"),
    ("CPP row-major vector", "cpp_rMajor_vector"),
    ("CPP column-major pointer", "cpp_cMajor_ptr"),
    ("CPP row-major pointer", "cpp_rMajor_ptr"),
    ("CPP gemm pointer", "cpp_gemm_ptr")
]

def parse_metrics_log(log_path):
    """Parse a structured metrics.log file into (epochs, dims, records).

    Returns:
        Tuple of lists: (epochs, dims, records)
        - epochs: List of epoch numbers.
        - dims: List of (M, N, K) tuples, one per epoch, in order.
        - records: List of metric dicts, one per epoch, shaped like:
            {
                "py_numpy" : <float seconds>,
                "py_loop" : <float seconds>,
                "cpp_rMajor_vector" : {"total_s": <float>, "alg_ms": <float>},
                "cpp_cMajor_ptr" : {"total_s": <float>, "alg_ms": <float>},
                "cpp_rMajor_ptr" : {"total_s": <float>, "alg_ms": <float>},
                "cpp_gemm_ptr"   : {"total_s": <float>, "alg_ms": <float>}
            }

    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    def parse_cpp(input_text, search_name):
        m = re.search(
            rf"{search_name}(?:(?!Method).)*?Total\(([\d.]+)s\)\s*\|\s*CPP Alg\(([\d.]+)ms\)",
            input_text, re.DOTALL
        )
        if m:
            return {"total_s": float(m.group(1)), "alg_ms": float(m.group(2))}
        else:
            return None

    # epoch_blocks alternates: [epoch_num, block_text, epoch_num, block_text, ...]
    epoch_blocks = re.split(r"={10}\s*Epoch\s+(\d+)\s*={10}", text)[1:]
    
    epochs = []
    dims = []
    records = []

    for i in range(0, len(epoch_blocks), 2):
        e = epoch_blocks[i]
        epochs.append(e)

        block_text = epoch_blocks[i + 1]

        dim_match = re.search(
            r"Dimensions:\s*C\((\d+),\s*(\d+)\)\s*=\s*A\((\d+),\s*(\d+)\)\s*x\s*B\((\d+),\s*(\d+)\)",
            block_text, re.DOTALL
        )
        m, n, _m, k, _k, _n = map(int, dim_match.groups())
        dims.append((m, n, k))

        record = {}

        # numpy matmul
        numpy_match = re.search(
            r"Numpy matmul.*?Duration:\s*([\d.]+)s",
            block_text, re.DOTALL
        )
        record["py_numpy"] = float(numpy_match.group(1))

        # python for-loop
        pyLoop_match = re.search(
            r"Python\s*for-loop(?:(?!Method).)*?Duration:\s*([\d.]+)s",
            block_text, re.DOTALL
        )
        if pyLoop_match:
            record["py_loop"] = float(pyLoop_match.group(1))
        else:
            record["py_loop"] = None

        # cpp core
        for name, key in METHODS[2:]:
            record[key] = parse_cpp(block_text, name)

        records.append(record)

    return epochs, dims, records



if __name__ == "__main__":
    try:
        log_path = sys.argv[1] if len(sys.argv) > 1 else Path("log", "metrics.log")

        for e, d, r in zip(*parse_metrics_log(log_path)):
            print(e, d, r)
    
    except Exception as e:
        print(e, file=sys.stderr)
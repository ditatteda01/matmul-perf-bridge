import sys
import os
import traceback
import numpy as np
import time
import logging
import random
from pathlib import Path


DTYPE = np.float32  # int32 / float32
REPETITION = 5      # Repeat algorithm whithin each epoch
EPOCH = 7
DIM_GROWTH_RATE = 2
MATRIX_DIM_M = 100   # Rows and columns of matrix multiplication:
MATRIX_DIM_K = 100   # C(M, N) = A(M, K) x B(K, N)
MATRIX_DIM_N = 100


def set_logger(log_dir: str, log_name: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formater_s = logging.Formatter("%(message)s")
    formater_f = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='w', encoding="utf-8")
        file_handler.setFormatter(formater_f)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formater_s)
        logger.addHandler(stream_handler)
    
    return logger

def check_matmul_result(mat_gt: np.ndarray, mat_test: np.ndarray) -> bool:
    """Checks whether input matrices the same. 

    Args:
        mat_gt: The ground truth matrix.
        mat_test: The testing matrix.

    Returns:
        True when the test matrix equals ground truth matrix.
    """
    return np.allclose(mat_gt, mat_test, rtol=1e-5, atol=1e-6)

def mk_matrices(rng: np.random.Generator, M: int, N: int, K: int, dtype: type) -> tuple[np.ndarray, np.ndarray]:
    """Create and return ndarray matrices.
    
    Args:
        rng: random generator.
        M: Rows of A.
        N: Columns of B.
        K: Columns of A and rows of B.
        dtype: Data type of matrix.

    Returns:
        (A, B).
    """
    if DTYPE == np.int32:
        A = rng.integers(-10, 10, size=(M, K), dtype=dtype)
        B = rng.integers(-10, 10, size=(K, N), dtype=dtype)

    elif DTYPE == np.float32:
        A = rng.random(size=(M, K), dtype=dtype)
        B = rng.random(size=(K, N), dtype=dtype)

    else:
        print(f"CRITICLE: Non supported data type (only int32/float32 available)", file=sys.stderr)
        sys.exit(1)

    return A, B

def matmul_python_loop(mat_mk: np.ndarray, mat_kn: np.ndarray) -> None:
    M, K = mat_mk.shape
    N = mat_kn.shape[1]
    mat_mn = np.zeros((M, N), dtype=mat_mk.dtype)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                mat_mn[i][j] += mat_mk[i][k] * mat_kn[k][j]
    return mat_mn

def warm_up(mthd: callable, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return mthd(A, B)

def time_alg(mthd: callable, mthdName: str, A: np.array, B: np.array) -> tuple:
    """Time a matrix multiplication algorithm and return results.
    
    Args:
        mthd: Matrix multiplication method to benchmark.
        mthdName: Name of the method.
        A: Left operand matrix.
        B: Right operand matrix.
    
    Returns:
        tuple:
            - For Numpy, Python methods: ``(result_matrix, wall_time_sec)``
            - For C++ methods: ``(result_matrix, wall_time_sec, algorithm_time_ms)``
    """
    if mthdName.startswith("CPP"):
        st = time.perf_counter()
        alg_ms, C = mthd(A, B)
        ed = time.perf_counter()
        result = (C, ed - st, alg_ms)
    else:
        st = time.perf_counter()
        C = mthd(A, B)
        ed = time.perf_counter()
        result = (C, ed - st)
    return result

def get_median(records: list, tgt_idx: int) -> tuple | None:
    """Return the median of the input on target index.
    
    Args:
        records: List of tuples.
        tgt_idx: Comparing index of the tuple.
    
    Returns:
        The median element of the input. Or `None` if `None` is found inside the input.
    """
    if None in records:
        return None

    n = len(records)
    records.sort(key=lambda x: x[tgt_idx])
    return records[n//2]

if __name__ == "__main__":

    # Add cpp module path to the system
    sys.path.append(os.path.abspath("./build"))

    try:
        logger = set_logger("./log", "metrics")

    except Exception as e:
        # write the error message to the terminal through stderr stream
        print(f"CRITICLE: Failed to initialize logger: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        import matmul_cpp as mmcpp
        logger.info("CPP module import correctly")
        logger.info(f"{REPETITION} reps for each Algorithm and the median is selected.")

        ALGORITHMS = [
            (np.matmul, "Numpy matmul"),
            (matmul_python_loop, "Python for-loop"),
            (mmcpp.matmul_rMajor_vector, "CPP row-major vector"),
            (mmcpp.matmul_cMajor_ptr, "CPP column-major pointer"),
            (mmcpp.matmul_rMajor_ptr, "CPP row-major pointer"),
            (mmcpp.gemm_f32_kernel4x16, "CPP gemm pointer")
        ]
        ALG_INDICES = [i for i in range(len(ALGORITHMS))]
        hasWarmup = False
        seed = 0
        random.seed(seed)
        rng = np.random.default_rng(seed=seed)

        for epoch in range(EPOCH):
            logger.info(f"====================== Epoch {epoch} ======================")

            M, N, K = map(
                lambda x: x * (DIM_GROWTH_RATE**epoch),
                [MATRIX_DIM_M, MATRIX_DIM_N, MATRIX_DIM_K]
            )
            A, B = mk_matrices(rng, M, N, K, DTYPE)
            C = np.matmul(A, B)
            logger.info(f"Dimensions: C{C.shape} = A{A.shape} x B{B.shape}")
            logger.info(f"Data Type: {DTYPE}")

            if not hasWarmup:
                for mthd, _ in ALGORITHMS:
                    warm_up(mthd, A, B)
                logger.info("Warm up algorithms Complete.")
                hasWarmup = True

            logger.info("Start benchmark...")
            logger.info("-"*54)

            alg_perfs = {name: [] for _, name in ALGORITHMS}
            for rep in range(REPETITION):
                random.shuffle(ALG_INDICES)

                for idx in ALG_INDICES:
                    mthd, mthdName = ALGORITHMS[idx]

                    # Skip Python and CPP column-major for big matrix
                    if epoch > 3 and (mthdName.startswith("Python") or mthdName.find("column") > -1):
                        perf = None
                    else:
                        perf = time_alg(mthd, mthdName, A, B)

                    alg_perfs[mthdName].append(perf)

                    if perf:
                        print(f"{mthdName} done | Wall time: {perf[1]:.6f}")
                    else:
                        print(f"{mthdName} skipped")
                
                print("~"*54)

            for name, perfs in alg_perfs.items():
                logger.info(f"Method: {name}")

                mperf = get_median(perfs, 1)
                if mperf is None:
                    logger.info("Status: Skipped (Matrix too large).")

                else:
                    logger.info("Status: Complete.")

                    if name.startswith("Numpy") or name.startswith("Python"):
                        logger.info(f"Duration: {mperf[1]:.6f}s")

                    elif name.startswith("CPP"):
                        logger.info(f"Duration: Total({mperf[1]:.6f}s) | CPP Alg({mperf[2]:.6f}ms)")

                    logger.info(f"Result Check: {check_matmul_result(C, mperf[0])}")

                logger.info("-"*54)

        logger.info("Benchmark Complete.")

    except ImportError as e:
        logger.critical(f"Import CPP module error: {e}")

    except Exception as e:
        logger.exception(f"Runtime error: {e}")
import sys
import os
import traceback
import numpy as np
import time
import logging
from pathlib import Path


DTYPE = np.float64
REPETITION = 5      # Repeat algorithm whithin each epoch
EPOCH = 7
DIM_GROWTH_RATIO = 2
MATRIX_DIM_P = 50   # Rows and columns of matrix multiplication:
MATRIX_DIM_Q = 30   # C(p, r) = A(p, q) x B(q, r)
MATRIX_DIM_R = 10


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
    return np.allclose(mat_gt, mat_test, 0)

def mk_matrices(p: int, q: int, r: int, dtype: type) -> tuple[np.ndarray, np.ndarray]:
    """Create and return ndarray matrices.
    
    Args:
        p: Rows of A.
        q: Columns of A and rows of B.
        r: Columns of B.
        dtype: Data type of matrix.

    Returns:
        (A, B).
    """
    rng = np.random.default_rng()
    if DTYPE == np.int32:
        A = rng.integers(-10, 10, size=(p, q), dtype=dtype)
        B = rng.integers(-10, 10, size=(q, r), dtype=dtype)
    elif DTYPE == np.float64:
        A = rng.random(size=(p, q), dtype=dtype)
        B = rng.random(size=(q, r), dtype=dtype)
    return A, B

def matmul_python_loop(mat_pq: np.ndarray, mat_qr: np.ndarray) -> None:
    p, q = mat_pq.shape
    r = mat_qr.shape[1]
    mat_pr = np.zeros((p, r))
    for i in range(p):
        for j in range(r):
            for k in range(q):
                mat_pr[i][j] += mat_pq[i][k] * mat_qr[k][j]
    return mat_pr

def warm_up(mthd: callable, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A warm-up call for matrix multiplication methods.

    Args:
        mthd: The matrix multiplication method.
        A: The left matrix of the multiplication.
        B: The right matrix of the multiplication.

    Returns:
        The matrix of A x B.
    """
    return mthd(A, B)

def rep_alg(mthd: callable, mthdName: str, A: np.ndarray, B: np.ndarray, reps=5) -> tuple:
    """Execute a matrix multiplication algorithm multiple times and return the median run.
    
    Args:
        mthd: Matrix multiplication method to benchmark.
        mthdName: Name of the method.
        A: Left operand matrix.
        B: Right operand matrix.
        reps: Number of repeated executions.
    
    Returns:
        tuple:
            - For Python methods: ``(result_matrix, wall_time_sec)``
            - For C++ methods: ``(result_matrix, wall_time_sec, algorithm_time_ms)``
    """
    perfs = []
    for _ in range(reps):
        if mthdName.startswith("CPP"):
            st = time.perf_counter()
            alg_ms, C = mthd(A, B)
            ed = time.perf_counter()
            perfs.append((C, ed - st, alg_ms))
        else:
            st = time.perf_counter()
            C = mthd(A, B)
            ed = time.perf_counter()
            perfs.append((C, ed - st))

    perfs.sort(key=lambda pair: pair[1])
    return perfs[reps // 2]


if __name__ == "__main__":

    # Add cpp module path to the system
    sys.path.append(os.path.abspath("./build"))

    try:
        logger = set_logger("./docs", "metrics")

    except Exception as e:
        # write the error message to the terminal through stderr stream
        print(f"CRITICLE: Failed to initialize logger: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        import matmul_cpp as mmcpp

        logger.info("CPP module import correctly")
        logger.info(f"Data Type: {DTYPE}")

        logger.info("Warm up algorithms...")

        ALGORITHMS = [
            (np.matmul, "Numpy matmul"),
            (matmul_python_loop, "Python for-loop"),
            (mmcpp.matmul_rMajor_vector, "CPP row-major vector"),
            (mmcpp.matmul_cMajor_ptr, "CPP column-major pointer"),
            (mmcpp.matmul_rMajor_ptr, "CPP row-major pointer"),
            # TODO: add optimization methods
        ]

        A, B = mk_matrices(MATRIX_DIM_P, MATRIX_DIM_Q, MATRIX_DIM_R, DTYPE)
        for fnc, _ in ALGORITHMS:
            warm_up(fnc, A, B)

        logger.info("Done.")
        logger.info("Start matrix multiplication: C = A x B")

        for epoch in range(EPOCH):
            logger.info(f"========== Epoch {epoch} ==========")

            p = MATRIX_DIM_P * DIM_GROWTH_RATIO**(epoch)
            q = MATRIX_DIM_Q * DIM_GROWTH_RATIO**(epoch)
            r = MATRIX_DIM_R * DIM_GROWTH_RATIO**(epoch)

            A, B = mk_matrices(p, q, r, DTYPE)
            logger.info(f"Dimensions: A{A.shape} x B{B.shape}")

            for mthd, name in ALGORITHMS:
                if name.startswith("Numpy"):
                    logger.info(f"Method: {name} | Status: Running {REPETITION}-times...")
                    GroundTruth, total_s = rep_alg(mthd, name, A, B, REPETITION)
                    logger.info(f"Done. Matrix C{GroundTruth.shape}. Duration: {total_s:.6f}s")

                elif name.startswith("Python"):
                    if (p * q * r < 100000000):
                        logger.info(f"Method: {name} | Status: Running {REPETITION}-times...")
                        C_python_loop, total_s = rep_alg(mthd, name, A, B, REPETITION)
                        logger.info(f"Done. Duration: {total_s:.6f}s")
                        logger.info(f"Result Check: {check_matmul_result(GroundTruth, C_python_loop)}")
                    else:
                        logger.info(f"Method: {name} | Status: Skipped (Matrix too large).")

                elif name.startswith("CPP"):
                    logger.info(f"Method: {name} | Status: Running {REPETITION}-times...")
                    C_cpp_xx, total_s, alg_ms = rep_alg(mthd, name, A, B, REPETITION)
                    logger.info(f"Done. Duration: Total({total_s:.6f}s) | CPP Alg({alg_ms:.6f}ms)")
                    logger.info(f"Result Check: {check_matmul_result(GroundTruth, C_cpp_xx)}")
                
                logger.info("-"*30)

        logger.info("Process terminated.")

    except ImportError as e:
        logger.critical(f"Import CPP module error: {e}")

    except Exception as e:
        logger.exception(f"Runtime error: {e}")
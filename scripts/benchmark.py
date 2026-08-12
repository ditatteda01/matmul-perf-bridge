import sys
import os
import traceback
import numpy as np
import time
import logging
from pathlib import Path


DTYPE = np.float32  # int32 / float32
REPETITION = 5      # Repeat algorithm whithin each epoch
EPOCH = 8
DIM_GROWTH_RATIO = 2
MATRIX_DIM_M = 50   # Rows and columns of matrix multiplication:
MATRIX_DIM_K = 30   # C(M, N) = A(M, K) x B(K, N)
MATRIX_DIM_N = 40


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

def mk_matrices(M: int, K: int, N: int, dtype: type) -> tuple[np.ndarray, np.ndarray]:
    """Create and return ndarray matrices.
    
    Args:
        M: Rows of A.
        K: Columns of A and rows of B.
        N: Columns of B.
        dtype: Data type of matrix.

    Returns:
        (A, B).
    """
    rng = np.random.default_rng()
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
            (mmcpp.gemm_f32_kernel4x16, "CPP gemm single-thread pointer"),
            # TODO: add optimization methods
        ]

        A, B = mk_matrices(MATRIX_DIM_M, MATRIX_DIM_K, MATRIX_DIM_N, DTYPE)
        for fnc, _ in ALGORITHMS:
            warm_up(fnc, A, B)

        logger.info("Done.")
        logger.info("Start matrix multiplication: C = A x B")

        for epoch in range(EPOCH):
            logger.info(f"====================== Epoch {epoch} ======================")

            M = MATRIX_DIM_M * DIM_GROWTH_RATIO**(epoch)
            K = MATRIX_DIM_K * DIM_GROWTH_RATIO**(epoch)
            N = MATRIX_DIM_N * DIM_GROWTH_RATIO**(epoch)

            A, B = mk_matrices(M, K, N, DTYPE)
            logger.info(f"Dimensions: A{A.shape} x B{B.shape}")

            for mthd, name in ALGORITHMS:
                if name.startswith("Numpy"):
                    logger.info(f"Method: {name} | Status: Running...")
                    GroundTruth, total_s = rep_alg(mthd, name, A, B, REPETITION)
                    logger.info(f"Done. Matrix C{GroundTruth.shape}. Duration: {total_s:.6f}s")

                elif name.startswith("Python"):
                    if (M * K * N < 100000000):
                        logger.info(f"Method: {name} | Status: Running...")
                        C_python_loop, total_s = rep_alg(mthd, name, A, B, REPETITION)
                        logger.info(f"Done. Duration: {total_s:.6f}s")
                        logger.info(f"Result Check: {check_matmul_result(GroundTruth, C_python_loop)}")
                    else:
                        logger.info(f"Method: {name} | Status: Skipped (Matrix too large).")

                elif name.startswith("CPP"):
                    if name.find("column") > -1 and M * K * N >= 10000000000:
                        logger.info(f"Method: {name} | Status: Skipped (Matrix too large).")
                    else:
                        logger.info(f"Method: {name} | Status: Running...")
                        C_cpp_xx, total_s, alg_ms = rep_alg(mthd, name, A, B, REPETITION)
                        logger.info(f"Done. Duration: Total({total_s:.6f}s) | CPP Alg({alg_ms:.6f}ms)")
                        logger.info(f"Result Check: {check_matmul_result(GroundTruth, C_cpp_xx)}")
                
                logger.info("-"*54)

        logger.info("Benchmark Complete.")

    except ImportError as e:
        logger.critical(f"Import CPP module error: {e}")

    except Exception as e:
        logger.exception(f"Runtime error: {e}")
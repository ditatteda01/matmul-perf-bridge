import sys
import os

# Add the build directory to sys.path to ensure the C++ module can be found
sys.path.append(os.path.abspath("./build"))

if __name__ == "__main__":
    try:
        import matmul_cpp as mmc
        print("C++ module loaded successfully.")
        print(f"Testing greet function: {mmc.greet('World')}")




    except ImportError as e:
        print("Failed to load C++ module:", e)
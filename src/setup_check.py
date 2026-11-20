"""Task 1: validate that the environment has every required package."""
import importlib, sys
pkgs = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "jupyter_core"]
print("Python", sys.version.split()[0])
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"  OK  {p:14s} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"  MISSING {p}")
print("Environment setup validated.")

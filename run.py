"""Double-click or run this file to execute the sample-only Groq pipeline."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    sys.argv = [str(project_root / "code" / "main.py"), "--input", "sample", "--llm"]
    runpy.run_path(project_root / "code" / "main.py", run_name="__main__")
    sys.argv = [str(project_root / "code" / "evaluation" / "main.py")]
    runpy.run_path(project_root / "code" / "evaluation" / "main.py", run_name="__main__")

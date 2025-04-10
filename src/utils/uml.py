import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_python_files(project_dir: Path) -> list[Path]:
    """
    Recursively finds all '.py' files within the project directory.
    Returns paths relative to the project directory.
    """
    abs_project_dir = project_dir.resolve()
    py_files = [p.relative_to(abs_project_dir) for p in abs_project_dir.rglob("*.py")]
    return py_files


def run_pyreverse(project_dir: Path, file_paths: list[Path], output_dir: Path):
    """
    Constructs and runs the pyreverse command targeting specific .py files.
    """
    pyreverse_cmd = shutil.which("pyreverse")
    if not pyreverse_cmd:
        print("Error: 'pyreverse' command not found.", file=sys.stderr)
        print(
            "Hint: Ensure pylint is installed ('pip install pylint') and pyreverse is in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for pyreverse - use relative paths for files
    file_args = [str(p) for p in file_paths]

    # Check if the command might be too long (simple heuristic)
    # Max command length varies by OS, but ~32k chars on Windows, often more on Linux
    # Let's warn if total length seems excessive (e.g., > 8000 chars)
    command_base_len = (
        len(pyreverse_cmd)
        + len("--output=puml")
        + len("-d")
        + len(str(output_dir.resolve()))
        + 20
    )  # Base length + spaces
    args_len = sum(len(arg) + 1 for arg in file_args)  # Length of args + spaces
    if command_base_len + args_len > 8000:
        print(
            f"[*] Warning: Found {len(file_args)} Python files. The generated command might be very long, potentially exceeding system limits.",
            file=sys.stderr,
        )

    command = [
        pyreverse_cmd,
        "--output=puml",
        "-d",
        str(output_dir.resolve()),
    ] + file_args

    print(f"[*] Running pyreverse on {len(file_args)} Python file(s).")
    print(f"[*] Output directory: {output_dir.resolve()}")
    # Avoid printing extremely long commands
    if command_base_len + args_len < 2000:
        print(f"[*] Command: {' '.join(command)}")
    else:
        print(
            f"[*] Command: {command[0]} --output=puml -d {output_dir.resolve()} [ ... {len(file_args)} files ... ]"
        )

    try:
        # Run pyreverse from the project directory
        result = subprocess.run(
            command,
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            errors="ignore",
        )
        print("[+] Pyreverse ran successfully.")
        print(f"[*] Output files should be in: {output_dir.resolve()}")
        # Optional: Print output if needed for debugging, can be noisy
        # print(result.stdout)
        # print(result.stderr)

    except FileNotFoundError:
        print(
            f"Error: Failed to run '{pyreverse_cmd}'. Ensure pylint is installed and in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"Error: Pyreverse failed with exit code {e.returncode}.", file=sys.stderr
        )
        print("--- Pyreverse stdout ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("--- Pyreverse stderr ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        # Handle potential "Argument list too long" error
        if e.errno == 7:  # errno.E2BIG
            print(
                f"Error: The command with {len(file_args)} files is too long for the operating system.",
                file=sys.stderr,
            )
            print(
                "Hint: Consider running pyreverse on smaller subsets of files or structuring your project into packages.",
                file=sys.stderr,
            )
        else:
            print(f"An OS error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan a Python project directory, find all .py files, "
        "and generate PlantUML class diagrams using pyreverse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "project_dir", type=str, help="Path to the Python project directory to scan."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="puml_output",
        help="Directory where the PlantUML (.puml) files will be saved.",
    )
    # TODO: Add arguments for excluding specific files/directories if needed
    # parser.add_argument("-e", "--exclude", action='append', help="Patterns to exclude")

    args = parser.parse_args()

    project_path = Path(args.project_dir)
    output_path = Path(args.output_dir)

    if not project_path.is_dir():
        print(
            f"Error: Project directory '{project_path}' not found or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find all .py files relative to the project path
    py_file_paths = find_python_files(project_path)

    if not py_file_paths:
        print(
            f"[*] Error: No Python files ('.py') found recursively within '{project_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_pyreverse(project_path, py_file_paths, output_path)
    print("[*] Script finished.")


if __name__ == "__main__":
    main()

"""
Windows EXE Launcher for Research Paper Explainer.

This script:
1. Starts the Streamlit server in the background.
2. Opens the browser automatically.
3. Waits for the user to close the browser/terminal.

Build with: pyinstaller launcher.spec
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal

PORT = 8502
URL = f"http://localhost:{PORT}"


def get_app_dir():
    """Get the directory where the app files are located."""
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))


def main():
    app_dir = get_app_dir()
    app_py = os.path.join(app_dir, "app.py")

    if not os.path.exists(app_py):
        print(f"ERROR: Cannot find app.py at {app_py}")
        input("Press Enter to exit...")
        sys.exit(1)

    print("=" * 50)
    print("  Research Paper Explainer")
    print("  Starting server...")
    print("=" * 50)

    # Start Streamlit as a subprocess
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(PORT)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_py,
         "--server.port", str(PORT),
         "--server.headless", "true"],
        cwd=app_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to start
    print(f"  Waiting for server on port {PORT}...")
    time.sleep(3)

    # Open browser
    print(f"  Opening {URL}")
    webbrowser.open(URL)

    print(f"\n  App is running at {URL}")
    print("  Press Ctrl+C to stop.\n")

    try:
        # Keep running until user presses Ctrl+C
        while process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("  Goodbye!")


if __name__ == "__main__":
    main()

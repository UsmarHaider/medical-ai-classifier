"""
Quick start script to run the Streamlit application
"""
import os
import sys
import subprocess

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add to path
sys.path.insert(0, PROJECT_ROOT)

def main():
    """Run the Streamlit application."""
    app_path = os.path.join(PROJECT_ROOT, "streamlit_app", "app.py")

    print("="*60)
    print("Medical Image Classification System")
    print("="*60)
    print(f"\nStarting Streamlit application...")
    print(f"App path: {app_path}")
    print("\nThe application will open in your web browser.")
    print("Press Ctrl+C to stop the server.\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except FileNotFoundError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()

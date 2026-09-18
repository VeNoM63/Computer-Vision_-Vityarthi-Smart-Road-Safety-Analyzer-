import os
import subprocess

if __name__ == "__main__":
    app_path = os.path.join("app", "app.py")
    subprocess.run(["streamlit", "run", app_path])

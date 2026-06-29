# Kaggle SCRIPT kernel — data acquisition (internet ON, GPU off).
# Pushed via: kaggle kernels push -p ./kaggle/   (after editing kernel-metadata.json)
import os, subprocess
REPO = "https://github.com/<you>/solar-ar-forecast.git"
subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
os.chdir("solar-ar-forecast")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
# JSOC email comes from a Kaggle Secret in the real run; set env here:
# from kaggle_secrets import UserSecretsClient
# os.environ["JSOC_EMAIL"] = UserSecretsClient().get_secret("JSOC_EMAIL")
subprocess.run(["python", "-m", "src.data.fetch", "--config", "configs/data_v1.yaml"], check=True)
# After this finishes: "Save Version" -> output becomes Dataset solar-event-windows-v1

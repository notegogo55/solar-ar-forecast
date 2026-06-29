# Kaggle SCRIPT kernel — segmentation training (GPU ON, internet ON).
# Attach datasets: solar-event-windows-v1 (data) + solar-ckpts (resume).
import os, subprocess
REPO = "https://github.com/<you>/solar-ar-forecast.git"
CONFIG = "configs/unet.yaml"   # or configs/surya.yaml
subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
os.chdir("solar-ar-forecast")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("commit:", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())
subprocess.run(["python", "-m", "src.seg.train", "--config", CONFIG], check=True)
# Outputs -> /kaggle/working; "Save Version" persists them as solar-ckpts.

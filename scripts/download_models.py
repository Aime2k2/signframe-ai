"""Download the original CNN checkpoint and MediaPipe asset for the desktop app."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    ("models/feedback_candidate.pth",
     "https://raw.githubusercontent.com/tayyabastack/signframe-ai/main/models/feedback_candidate.pth",
     "8fdf970ef3356709e14023fa89b186cc3ecf6a466ccaf8a06807bc2462c6388f"),
    ("assets/hand_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
     None),
]

for path, url, sha in ASSETS:
    dest = ROOT / path
    if dest.is_file() and (not sha or hashlib.sha256(dest.read_bytes()).hexdigest() == sha):
        print("Present:", path)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading:", path)
    with urlopen(url, timeout=120) as response:
        data = response.read()
    if sha and hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError("Unexpected SHA256 for " + path)
    dest.write_bytes(data)
    print("Saved:", path, len(data), "bytes")

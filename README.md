# SignFrame AI

**Standalone portfolio package finalized:** 15 November 2026  
**Portfolio release and browser adaptation:** Jean Aimé Iraguha ([Aime2k2](https://github.com/Aime2k2))  
**Original Python implementation:** Tayyaba Ilyas ([upstream SignFrame AI](https://github.com/tayyabastack/signframe-ai))

SignFrame AI is an experimental webcam-based American Sign Language (ASL) fingerspelling recognizer. It detects hands with MediaPipe, uses a PyTorch convolutional neural network to classify 24 static letters (excluding J and Z), and helps users compose and speak text.

## Website

A browser demo lives in `web/`. Deployment is configured through `.github/workflows/deploy-pages.yml`. Once GitHub Pages is set to **GitHub Actions** and a deployment succeeds, visit:

https://aime2k2.github.io/signframe-ai/

This address is not guaranteed to be live until deployment succeeds.

## Run the original desktop app (Windows, Python 3.12)

```powershell
git clone https://github.com/Aime2k2/signframe-ai.git
cd signframe-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_models.py
python -m src.webcam_app
```

Allow webcam access when prompted. The Python desktop application has not been independently webcam-tested in this environment.

## Run the browser demo locally

```bash
python -m http.server 8000
```

Open http://localhost:8000/web/ (not `file://`), select **Start camera**, and approve access. Internet is required for MediaPipe runtime assets. This is an experimental demonstration, and recognition accuracy for new signers has not been independently established.

## Model

The trained checkpoint comes from the original author. The site build downloads the source checkpoint and converts its tensor storage to a Float32 browser format. We do not retrain it here.

## Attribution and permissions

This repository preserves attribution to [Tayyaba Ilyas's original implementation](https://github.com/tayyabastack/signframe-ai). The original repository does not contain an explicit license. Before distributing, reusing, or relicensing original files, obtain appropriate permission from the author. The November 2026 packaging and web adaptation do not imply independent authorship of the upstream CNN or historical evaluations.

**Research prototype:** Not intended for critical communications. This is letter recognition, not full ASL translation.

# Deployment guide (15 November 2026)

The website is a **GitHub Pages static app**, not the original Python GUI. It runs webcam capture, MediaPipe hand detection, and CNN letter inference in the browser. Models run locally once loaded. It requires a modern HTTPS browser.

## Publish

1. The repo is https://github.com/Aime2k2/signframe-ai (branch `main`).
2. In GitHub, open **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. Under **Actions**, open **Deploy SignFrame AI web demo**. Wait for its run to succeed. If needed, use **Run workflow**.
4. Open https://aime2k2.github.io/signframe-ai/ and click **Start camera**.
5. Permit camera access; show an individual static ASL sign. Add stable predictions to the message and use the speech button.

## Model packaging

The workflow runs `python3 scripts/build_web_weights.py`, which downloads the checkpoint from upstream, verifies its SHA-256, and extracts the eight unchanged float32 tensor storages to `web/model/weights.bin`. The binary is an **artifact**, not tracked by Git, and the website deploy includes it automatically.

## Troubleshooting

- **Pages setup error:** Settings → Pages → select GitHub Actions, then rerun the deployment workflow.
- **Permission error:** Check repository Actions permissions and environment deployment rules.
- **Blank model/loading error:** Check the Actions build and that `web/model/weights.bin` was generated. The model source is external and must be reachable by Actions.
- **Camera denied:** Use HTTPS or localhost; grant browser camera permission and close other camera apps.
- **MediaPipe not loading:** Check network access to jsDelivr and Google MediaPipe model hosting.
- **Poor recognition:** Light the hand evenly, use a simple background, and note that independent webcam accuracy has not been established.

Local browser testing from the repository root: `python -m http.server 8000`, then visit `http://localhost:8000/web/`. Run `python scripts/build_web_weights.py` first to generate the browser model.

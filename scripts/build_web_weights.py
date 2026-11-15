"""Convert the original PyTorch ZIP checkpoint to browser Float32 weights.

No PyTorch dependency is needed here: the checkpoint stores uncompressed little-
endian float32 tensor storages, and our exported CNN uses the same storage order.
The manifest at web/model/weights.json declares the expected tensors.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/tayyabastack/"
    "signframe-ai/main/models/feedback_candidate.pth"
)
CHECKPOINT_SHA256 = "8fdf970ef3356709e14023fa89b186cc3ecf6a466ccaf8a06807bc2462c6388f"
WEIGHTS_SHA256 = "79fae5c3e4190addcc325e1289b14e9a5a5f12b979d31efa7eab28d7dcbbbb92"

def main() -> None:
    manifest = json.loads((ROOT / "web/model/weights.json").read_text())
    if manifest["format"] != "signframe-cnn-f32-v1":
        raise ValueError("Unexpected model manifest format")
    print("Downloading original source checkpoint...")
    with urlopen(CHECKPOINT_URL, timeout=180) as response:
        checkpoint = response.read()
    actual_checkpoint = hashlib.sha256(checkpoint).hexdigest()
    if actual_checkpoint != CHECKPOINT_SHA256:
        raise ValueError("Upstream checkpoint changed; SHA256 verification failed")

    with ZipFile(io.BytesIO(checkpoint)) as archive:
        prefix = next(
            name.rsplit("data/0", 1)[0]
            for name in archive.namelist() if name.endswith("data/0")
        )
        weights = b"".join(archive.read(f"{prefix}data/{i}") for i in range(8))

    layer_sizes = [
        manifest["layers"][name]["count"]
        for name in (
            "features.0.weight", "features.0.bias",
            "features.3.weight", "features.3.bias",
            "classifier.1.weight", "classifier.1.bias",
            "classifier.4.weight", "classifier.4.bias",
        )
    ]
    if sum(layer_sizes) * 4 != len(weights):
        raise ValueError("Weights size mismatch")
    if hashlib.sha256(weights).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Exported weights did not match validated browser model")
    destination = ROOT / "web/model/weights.bin"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(weights)
    print(f"Verified and prepared {len(weights):,} bytes: {destination}")

if __name__ == "__main__":
    main()

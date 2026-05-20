"""SplatfastK1 — Replicate predictor wrapping Brush.

Accepts:
  - colmap_zip   (Path): a .zip containing a COLMAP-formatted dataset
                          (images/, sparse/0/cameras.bin, images.bin, points3D.bin)
  - total_steps  (int):  number of Brush training iterations (default 30000)

Returns:
  - scene.ply    (Path): the trained Gaussian splat
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath


BRUSH_EXE = "/opt/brush/brush_app"
WORK_DIR = Path("/tmp/sf_work")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Quick sanity check that the Brush binary is present at container boot."""
        if not Path(BRUSH_EXE).exists():
            raise RuntimeError(f"Brush binary not found at {BRUSH_EXE}")
        # Print version for the build logs
        try:
            result = subprocess.run(
                [BRUSH_EXE, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(f"Brush ready: {(result.stdout or result.stderr).strip()}")
        except Exception as e:
            print(f"Brush version check failed (non-fatal): {e}")

    def predict(
        self,
        colmap_zip: CogPath = Input(
            description="Zip file containing a COLMAP dataset (images/ and sparse/0/).",
        ),
        total_steps: int = Input(
            description="Number of Brush training iterations.",
            default=30000,
            ge=1000,
            le=50000,
        ),
    ) -> CogPath:
        """Run Brush on the COLMAP dataset and return the trained scene.ply."""

        # Fresh working dir per prediction
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
        WORK_DIR.mkdir(parents=True, exist_ok=True)

        dataset_dir = WORK_DIR / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Unzip the COLMAP dataset
        print(f"Unpacking {colmap_zip} to {dataset_dir}")
        with zipfile.ZipFile(str(colmap_zip), "r") as zf:
            zf.extractall(str(dataset_dir))

        # Brush expects images/ and sparse/ at the top of the dataset.
        # If the zip nested everything under a single top folder, flatten it.
        items = list(dataset_dir.iterdir())
        if len(items) == 1 and items[0].is_dir():
            inner = items[0]
            for child in inner.iterdir():
                shutil.move(str(child), str(dataset_dir / child.name))
            inner.rmdir()

        if not (dataset_dir / "images").exists():
            raise ValueError("Zip is missing 'images/' folder.")
        if not (dataset_dir / "sparse").exists():
            raise ValueError("Zip is missing 'sparse/' folder.")

        splat_dir = WORK_DIR / "splat"
        splat_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            BRUSH_EXE,
            str(dataset_dir),
            "--total-steps",
            str(total_steps),
            "--export-path",
            str(splat_dir),
            "--export-name",
            "scene.ply",
        ]
        print(f"Running: {' '.join(cmd)}")

        # Run with live output. raises on non-zero exit.
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Brush exited with code {result.returncode}")

        out_ply = splat_dir / "scene.ply"
        if not out_ply.exists():
            raise RuntimeError(f"Brush finished but {out_ply} was not produced.")

        print(f"Done. scene.ply = {out_ply.stat().st_size / 1024 / 1024:.1f} MB")
        return CogPath(str(out_ply))

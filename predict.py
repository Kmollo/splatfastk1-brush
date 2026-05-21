"""SplatfastK1 — Replicate predictor wrapping Brush v0.3.0.

Inputs:
  - colmap_zip   (Path): zip containing a COLMAP dataset (images/, sparse/0/...)
  - total_steps  (int):  Brush training iterations

Output:
  - scene.ply    (Path)

Notes:
  * v0.3.0 of Brush exports a .ply only on multiples of --export-every. We set
    --export-every == total_steps so an export happens exactly at the end.
  * All print() calls flush so cog streams logs in real time (otherwise the
    prediction looks hung).
  * Brush uses Vulkan via wgpu, not CUDA. The container needs Vulkan ICDs —
    handled in cog.yaml.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath


BRUSH_EXE = "/opt/brush/brush_app"
WORK_DIR = Path("/tmp/sf_work")


def log(msg: str) -> None:
    """Print and flush immediately so cog streams the line to Replicate."""
    print(msg, flush=True)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Sanity-check the Brush binary at container boot."""
        log(f"[setup] Brush binary at {BRUSH_EXE}")
        if not Path(BRUSH_EXE).exists():
            raise RuntimeError(f"Brush binary not found at {BRUSH_EXE}")
        # Print version
        try:
            r = subprocess.run(
                [BRUSH_EXE, "--version"],
                check=False, capture_output=True, text=True, timeout=10,
            )
            log(f"[setup] Brush version: {(r.stdout or r.stderr).strip()}")
        except Exception as e:
            log(f"[setup] version check failed (non-fatal): {e}")

        # Check that a Vulkan ICD is present so wgpu can find the GPU
        icd_paths = [
            "/usr/share/vulkan/icd.d",
            "/etc/vulkan/icd.d",
            "/usr/local/share/vulkan/icd.d",
        ]
        found_icds: list[str] = []
        for p in icd_paths:
            if Path(p).exists():
                for icd in Path(p).iterdir():
                    found_icds.append(str(icd))
        log(f"[setup] Vulkan ICDs found: {found_icds or '(none — Brush may not see a GPU!)'}")

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
        """Run Brush on the dataset and return the trained scene.ply."""

        # --- Fresh working dir ---
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        dataset_dir = WORK_DIR / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # --- Unzip ---
        log(f"[1/5] Unpacking {colmap_zip} -> {dataset_dir}")
        with zipfile.ZipFile(str(colmap_zip), "r") as zf:
            zf.extractall(str(dataset_dir))

        # Flatten if the zip nested everything under a single top folder
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

        n_images = sum(1 for _ in (dataset_dir / "images").rglob("*") if _.is_file())
        n_sparse = sum(1 for _ in (dataset_dir / "sparse").rglob("*") if _.is_file())
        log(f"[1/5] Dataset ready: {n_images} images, {n_sparse} sparse files")

        # --- Run Brush ---
        splat_dir = WORK_DIR / "splat"
        splat_dir.mkdir(parents=True, exist_ok=True)

        # v0.3.0 only exports at multiples of --export-every. Align it with
        # --total-steps so the .ply is written at the very end. Also drop
        # the {iter} templating in the default name.
        cmd = [
            BRUSH_EXE,
            str(dataset_dir),
            "--total-steps", str(total_steps),
            "--export-every", str(total_steps),
            "--export-path", str(splat_dir),
            "--export-name", "scene.ply",
        ]
        log(f"[2/5] Running Brush: {' '.join(cmd)}")

        # Stream Brush's stdout live (line-buffered) so we see training progress.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "RUST_LOG": "info"},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log(f"[brush] {line.rstrip()}")
        rc = proc.wait()
        log(f"[3/5] Brush exited with code {rc}")
        if rc != 0:
            raise RuntimeError(f"Brush exited with code {rc}")

        # --- Locate the .ply ---
        # v0.3.0 may have used the export_name verbatim or with templating.
        # Look for scene.ply first; fall back to any .ply in splat_dir.
        out_ply = splat_dir / "scene.ply"
        if not out_ply.exists():
            log(f"[4/5] {out_ply} not found, looking for any .ply in {splat_dir}")
            candidates = sorted(splat_dir.rglob("*.ply"))
            if not candidates:
                raise RuntimeError(
                    f"Brush finished but no .ply was produced in {splat_dir}. "
                    f"Contents: {list(splat_dir.iterdir())}"
                )
            out_ply = candidates[-1]
            log(f"[4/5] Using {out_ply}")

        size_mb = out_ply.stat().st_size / 1024 / 1024
        log(f"[5/5] Done. scene.ply = {size_mb:.1f} MB at {out_ply}")
        return CogPath(str(out_ply))

# splatfastk1-brush

Cog package that runs the [Brush](https://github.com/ArthurBrussee/brush) Gaussian-splat trainer on a Replicate GPU.

This repository is the backend half of **SplatfastK1** — its only job is to host
the splat training step on the cloud so the desktop app's "Train in the cloud"
option works.

## What it does

Given a zip of a COLMAP-formatted dataset (frames + sparse reconstruction), it
runs Brush for the requested number of iterations and returns the trained
`scene.ply`.

## How it's published

A push to `main` that touches `cog.yaml`, `predict.py`, or the workflow file
triggers `.github/workflows/publish.yml`, which:

1. Installs [cog](https://github.com/replicate/cog) on a GitHub Actions runner
2. Logs in to Replicate using the `REPLICATE_API_TOKEN` repo secret
3. Builds the model image and pushes it to `r8.im/kmollo/splatfastk1-brush`

The model is then callable at <https://replicate.com/kmollo/splatfastk1-brush>.

## Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `colmap_zip` | file | required | Zip of a COLMAP dataset (`images/`, `sparse/0/`) |
| `total_steps` | int | `30000` | Brush training iterations (1000–50000) |

## Output

Path to a `scene.ply` ready to load in any Gaussian splat viewer or Blender via
the BlendSplat node graph.

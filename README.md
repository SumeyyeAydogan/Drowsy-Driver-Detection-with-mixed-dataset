# DDD-Project

PyTorch-based project for binary classification of drowsiness (Drowsy vs NotDrowsy). This repository includes data loading, training, evaluation, Grad-CAM visualization, and model export utilities.

## Requirements
- Python 3.9+
- Pip

Installation:
```bash
pip install -r requirements.txt
```

## Dataset Layout
The default dataset directory is `dataset/`:
```
dataset/
  Drowsy/
    <...>.png
  NotDrowsy/
    <...>.png
```
The `dataset/` folder is large and is ignored by Git via `.gitignore`.

## Quick Start
The main entry point is `main.py`.

Train example:
```bash
python main.py --mode train \
  --data_dir dataset \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --img_size 224
```

Evaluate example:
```bash
python main.py --mode eval \
  --data_dir dataset \
  --checkpoint_path checkpoints/best.pt
```

Grad-CAM example:
```bash
python main.py --mode gradcam \
  --checkpoint_path checkpoints/best.pt \
  --image_path path/to/image.png \
  --output_dir gradcam_output
```

Export example (ONNX):
```bash
python main.py --mode export \
  --checkpoint_path checkpoints/best.pt \
  --export_format onnx \
  --output_path exports/model.onnx
```

Note: The exact arguments are defined inside `main.py`.

## Code Map
- `src/dataloader.py`: Data loading and splitting
- `src/model.py`: Model definition
- `src/train.py`: Training loop
- `src/evaluate.py`: Evaluation
- `src/gradcam.py` and `gradcam_simple.py`: Grad-CAM visualization tools
- `src/export.py`: Export (ONNX, etc.)
- `src/utils.py`, `src/callbacks.py`, `src/run_manager.py`: Utilities and run management

## Development
Using a virtual environment is recommended:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Git and GitHub
After initial setup, to connect to a new GitHub repository:
```bash
git init -b main
git add .
git commit -m "chore: initial commit"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
If using SSH, generate a key and add the public key to GitHub first.

## License
No license specified yet. Add a `LICENSE` file if needed.

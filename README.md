# Emotion recognition (live webcam)

This project provides training and a realtime webcam demo for the FER2013 dataset.

Quick steps:

1. Unzip your dataset archive into `data/fer2013/`. The FER CSV should be at `data/fer2013/fer2013.csv` or processed images in `data/fer2013/processed/`.

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Train a model (optional if you already have `models/best_model.pth`):

```bash
python train.py --data data/fer2013 --epochs 20 --batch 64 --save models/best_model.pth
```

4. Run live demo (press `q` to quit):

```bash
python realtime.py --model models/best_model.pth
```

Notes:
- If you already have `models/best_model.pth`, the demo will load it. If not, the demo will run with an untrained model (use training to get accurate predictions).
- The code expects faces detected with OpenCV Haar cascades and performs grayscale 48x48 preprocessing.

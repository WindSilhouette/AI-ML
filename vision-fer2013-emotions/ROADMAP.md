# ROADMAP

**Milestone 1 — Minimal E2E (1–2 hrs)**
- [ ] Add a few toy images into `data/sample/<label>/` (two labels are enough).
- [ ] `python -m src.train` prints train/val accuracy and saves `models/smallcnn.pt`.

**Milestone 2 — Real Dataset (FER-2013)**
- [ ] Download FER-2013; convert to folder format (per-class subdirs) or a PyTorch Dataset.
- [ ] Implement Albumentations pipeline for training-only augmentations.
- [ ] Fix random seeds; stratified split into train/val/test.

**Milestone 3 — Strong Baselines**
- [ ] Add ResNet18 fine-tuning; compare vs SmallCNN.
- [ ] Report accuracy/F1 and confusion matrix; save misclassified examples to `artifacts/`.

**Milestone 4 — Explainability & Report**
- [ ] Implement Grad-CAM heatmaps for correct/incorrect samples.
- [ ] Summarize experiments + figures in `REPORT.md`.

# DINOv3 — see and classify, in two notebooks

A tiny, beginner-friendly demo of Meta's [DINOv3](https://github.com/facebookresearch/dinov3) vision model. Two notebooks, no GPU heroics, no training the big model — just enough to *get the idea*.

## What is DINOv3?

DINOv3 is an **image encoder**. You give it a picture; it gives you back a list of 768 numbers (a "feature vector" or "embedding") that summarize what's in the picture.

The interesting part is how Meta trained it: they showed it **1.689 billion images** with **zero labels**. The model never saw a single "this is a cat" tag. Instead, it was trained to produce *the same* feature vector for the same image under random crops, flips, and color changes. After enough of that, it learned, on its own, that cats look like cats and trucks look like trucks — without ever being told.

The result is an encoder whose 768-dim space is so well-organized that:

- Two pictures of cats sit close to each other.
- Cats and dogs are closer to each other than cats and trucks.
- A simple linear separator can tell 10 CIFAR-10 classes apart with **98% accuracy**.

This repo shows both halves of that claim.

## What's in here

```
dino/
├── scripts/
│   ├── dinov3.ipynb            # notebook 1 — feature extraction
│   └── dinov3_classify.ipynb   # notebook 2 — classification on top
```

### Notebook 1 — `scripts/dinov3.ipynb`

**The "what does an embedding look like?" notebook.**

Loads the DINOv3 ViT-B/16 model, runs one image through it, and shows you what comes out:

- A **CLS embedding** of shape `(1, 768)` — one feature vector per image.
- **Patch tokens** of shape `(1, 196, 768)` — one feature vector per 16×16 image patch (14×14 patches at 224×224 = 196 patches).

The last cell does a fun visualization: it takes the 196 patch embeddings, runs PCA to squish 768 dimensions down to 3, and renders those 3 numbers as RGB. The result is a colorful 14×14 grid where regions with similar features get similar colors — a quick way to see that the model is "noticing" the structure of the image.

### Notebook 2 — `scripts/dinov3_classify.ipynb`

**The "let's actually classify something" notebook.**

The pipeline:

```
CIFAR-10 image  →  DINOv3 (frozen)  →  768-d vector  →  Linear(768 → 10)  →  prediction
                   never updated                        the only trained part
```

Steps the notebook walks through:

1. **Freeze the backbone.** Set `requires_grad = False` everywhere. DINOv3 will not learn anything new.
2. **Extract features once.** Push all 60,000 CIFAR-10 images through the backbone and save the resulting `(50000, 768)` and `(10000, 768)` tensors to disk. After this step, the backbone is never used again.
3. **Train a tiny head.** A single `nn.Linear(768, 10)` matrix learns to draw 10 hyperplanes in the 768-d space — one per CIFAR-10 class. Each epoch takes well under a second because we're only doing matrix multiplies on cached vectors.
4. **k-NN baseline.** A bonus cell skips training entirely: for each test image, it finds the 20 closest training embeddings by cosine similarity and lets them vote. Often within a couple of percent of the trained head — direct evidence that the embedding space is already organized by class.

### Why this is interesting

Training a vision model from scratch on CIFAR-10 with a ResNet typically gets you ~95% after hours of work. This repo trains a single linear layer for **two minutes** and lands at **~98%**. The hard work was already done by Meta on those 1.689 billion unlabeled images. You're just learning where to draw the lines in their feature space.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for Python deps.

```bash
git clone https://github.com/htansetiawan/dino
cd dino
uv sync                           # installs torch, torchvision, etc. from pyproject.toml
```

Download the ViT-B/16 weights (gated by Meta — request a signed URL at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and place them at:

```
weights/dinov3_vitb16_pretrain_lvd1689m.pth
```

Drop any image you like at `image.jpg` (any size, RGB or grayscale).

Run:

```bash
uv run jupyter lab
```

Open `scripts/dinov3.ipynb` first, then `scripts/dinov3_classify.ipynb`.

## What to try next

- **Your own data.** Replace the CIFAR-10 cell with `datasets.ImageFolder('path/to/data', transform=transform)` — folder-per-class structure (`data/cat/*.jpg`, `data/dog/*.jpg`). Everything downstream just works.
- **Bigger backbones.** Swap `dinov3_vitb16` for `dinov3_vitl16` (300M params) or `dinov3_vit7b16` (7B params). More compute, slightly higher accuracy, same recipe.
- **Different similarity metric.** The k-NN cell uses cosine similarity. Try L2 distance and see how it changes things.
- **Visualize what each class lights up on.** Use the patch tokens (not the CLS) to make per-class attention maps.

## Credit

All the heavy lifting is Meta's. This repo is just glue: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).

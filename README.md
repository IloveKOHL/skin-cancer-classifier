# Skin Cancer Classifier

Deep-learning classification of dermatoscopic skin lesion images on the HAM10000
dataset, built with PyTorch and [timm](https://github.com/huggingface/pytorch-image-models).

> ## ⚠️ Disclaimer
> **This is a learning and research project. It is NOT a medical device.**
> It does not provide a diagnosis and is not a substitute for a dermatologist.
> Do not use it to make any decision about your health or anyone else's.
> If a skin lesion worries you, see a doctor.

## Project goal

Train and honestly evaluate an image classifier for skin lesions, as a hands-on
project for learning the full ML workflow: leakage-free data splitting, handling
class imbalance, mixed-precision training on a consumer GPU (8 GB VRAM),
clinically meaningful metrics, and explainability.

- **Binary mode** (default): benign vs. malignant
- **Multiclass mode**: the 7 HAM10000 diagnoses, with a focus on melanoma recall
- **Model**: EfficientNet-B0 with ImageNet weights (transfer learning); any timm
  backbone can be used by changing one config line (`resnet50`, `convnext_tiny`, ...)

## Data and licenses

| Dataset | Use | License |
|---|---|---|
| [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (Tschandl et al., 2018) | training, validation, test | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - **non-commercial only** |

- The dataset is **not** part of this repository. See [data/README.md](data/README.md)
  for download steps, citation and the class definitions.
- Model weights trained on HAM10000 are derived from non-commercially licensed data.
  No weights are published here.
- The code in this repository is released under the [MIT License](LICENSE).
- Pretrained ImageNet backbones are downloaded by timm; their licenses are listed
  on the respective timm model cards.

## Installation

Requires Python 3.11 and, for training, an NVIDIA GPU (developed on an RTX 4060, 8 GB).

```bash
git clone https://github.com/IloveKOHL/skin-cancer-classifier.git
cd skin-cancer-classifier
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` on Linux/macOS), then install PyTorch **with CUDA first**
and the remaining packages afterwards:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

The last command should print `True`. Pick the CUDA build that matches your driver
on <https://pytorch.org/get-started/locally/> if `cu128` does not fit.

### Windows: use WSL2 if Smart App Control is on

Windows 11 Smart App Control blocks the unsigned DLLs of PyTorch and pandas
(`WinError 4551`, "an application control policy has blocked this file"). Instead of
switching that protection off, run the project in WSL2 (Ubuntu). The GPU is passed
through by the normal Windows NVIDIA driver. This project was developed that way.

```bash
# in PowerShell, once
wsl --install -d Ubuntu-24.04

# inside Ubuntu, once (Ubuntu 24.04 ships Python 3.12, so add 3.11)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.11 python3.11-venv

# inside Ubuntu, from the repository folder (e.g. /mnt/c/Users/<you>/.../skin-cancer-classifier)
bash scripts/setup_wsl.sh
source ~/.venvs/skin-cancer/bin/activate
```

Tip: reading thousands of images from the Windows drive (`/mnt/c/...`) is slow in
WSL2. Store the dataset in the Linux file system instead, e.g.
`python scripts/download_data.py --output ~/data/HAM10000`, and point `data.root`
and `data.metadata_csv` to it.

## Quick start

```bash
# 0. Optional: check the whole pipeline on synthetic images (about a minute)
python scripts/make_dummy_data.py --output data/dummy
python -m src.train --config configs/smoke.yaml
pytest

# 1. Download HAM10000 (about 2.8 GB, CC BY-NC 4.0)
python scripts/download_data.py

# 2. Train (binary mode, EfficientNet-B0)
python -m src.train --config configs/default.yaml

# 3. Evaluate the best checkpoint on the held-out test split
python -m src.evaluate --checkpoint runs/<run_name>/best.pt --split test

# 4. Grad-CAM heatmaps for test images
python -m src.gradcam --checkpoint runs/<run_name>/best.pt --num-samples 8

# 5. Interactive demo (pass several checkpoints to compare models side by side)
python app/demo.py --checkpoint runs/<run_name>/best.pt
python app/demo.py --checkpoint runs/<binary_run>/best.pt runs/<multiclass_run>/best.pt
```

Watch training curves with `tensorboard --logdir runs`.

### Changing the setup

All settings live in [configs/default.yaml](configs/default.yaml). Single values can
be overridden on the command line:

```bash
# 7-class mode, best checkpoint chosen by melanoma recall
python -m src.train --override task.mode=multiclass train.monitor=recall

# different backbone, larger images, smaller batch to fit 8 GB VRAM
python -m src.train --override model.name=convnext_tiny data.image_size=384 train.batch_size=12

# oversample rare classes instead of weighting the loss
python -m src.train --override train.imbalance=weighted_sampler
```

## How it works

| Topic | Approach |
|---|---|
| **Data leakage** | HAM10000 has several images per lesion. The train/val/test split (70/15/15) is done **per `lesion_id`**, stratified by diagnosis, with a fixed seed. A test asserts that no lesion appears in two splits. |
| **Class imbalance** | About two thirds of the images are nevi. Choose inverse-frequency class weights in the loss (`weighted_loss`, default) or a `WeightedRandomSampler`. |
| **Augmentation** | Horizontal/vertical flips, rotation, moderate colour jitter; normalisation with the statistics of the pretrained backbone. |
| **Training** | AdamW, linear warmup + cosine learning-rate decay, mixed precision (AMP), gradient clipping, early stopping, best checkpoint by validation ROC-AUC (or recall). |
| **Metrics** | Sensitivity, specificity, precision, F1, ROC-AUC, balanced accuracy, confusion matrix (PNG), classification report; per-class recall and melanoma recall/specificity in multiclass mode. |
| **Reproducibility** | Fixed seeds; every run folder stores its full config, the split assignment (`splits.csv`), TensorBoard logs and validation metrics. Weights & Biases is optional (`logging.wandb: true`). |
| **Explainability** | Grad-CAM on the last convolutional feature map ([src/gradcam.py](src/gradcam.py)). |

Why not just accuracy? A model that calls every lesion "benign" is right most of the
time on this dataset and useless in practice. For a screening-type task the share of
malignant lesions that are found (sensitivity) matters most, balanced against
specificity.

## Project structure

```
configs/     default.yaml (all hyperparameters), smoke.yaml (synthetic test run)
data/        empty - download instructions in data/README.md
scripts/     download_data.py, make_dummy_data.py
src/         dataset.py, model.py, train.py, evaluate.py, predict.py, gradcam.py, utils.py
app/         demo.py (Gradio)
notebooks/   01_data_exploration.ipynb
tests/       pytest: split leakage, dataset, model forward, metrics, end-to-end smoke test
```

## Results

Values are copied from `runs/<run_name>/eval_test/metrics.json`. Entries marked TBD
have not been measured yet.

Binary mode (benign vs. malignant), HAM10000 test split (lesion-level split, seed 42,
1,497 images: 1,201 benign / 296 malignant), default config, single run, decision
threshold 0.5, best checkpoint from epoch 12 (selected by validation ROC-AUC):

| Model | Image size | ROC-AUC | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 224 | 0.920 | 0.824 | 0.865 | 0.601 | 0.695 |

Confusion matrix (rows = true, columns = predicted):

| | pred. benign | pred. malignant |
|---|---|---|
| **benign** | 1039 | 162 |
| **malignant** | 52 | 244 |

Read these numbers with care: 52 of 296 malignant lesions (about 18 %) were missed
at this threshold, and 4 of 10 "malignant" predictions were false alarms. This is a
single run without confidence intervals.

Multiclass mode (7 classes), same test split (168 melanoma images), default config
with `task.mode=multiclass`, single run, best checkpoint from epoch 20 (selected by
validation macro ROC-AUC):

| Model | Image size | Macro ROC-AUC | Melanoma recall | Melanoma specificity | Macro F1 | Balanced accuracy |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 224 | 0.961 | 0.649 | 0.963 | 0.704 | 0.674 |

Recall per class:

| akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|
| 0.647 | 0.714 | 0.516 | 0.455 | 0.649 | 0.962 | 0.773 |

The high macro ROC-AUC hides a weak spot: with argmax decisions the model missed
59 of 168 melanomas (35 %), 44 of them classified as benign nevi. Accuracy (0.846) is
dominated by the nevus class. Small classes (`df`, `vasc`: 22 test images each) have
very uncertain estimates.

### Multiclass experiments: trying to raise melanoma recall

Same model, data split and seed; one run each. Only the listed settings differ.

| Imbalance handling | Checkpoint chosen by | Best epoch | Macro ROC-AUC | Melanoma recall | Melanoma specificity | Macro F1 | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| `weighted_loss` (default) | val ROC-AUC | 20 | 0.961 | 0.649 | 0.963 | 0.704 | 0.674 |
| `weighted_loss` | val melanoma recall | 3 | 0.939 | 0.768 | 0.828 | 0.585 | 0.637 |
| `weighted_sampler` | val ROC-AUC | 22 | 0.961 | 0.643 | 0.947 | 0.731 | 0.723 |

- Selecting the checkpoint by melanoma recall picked an early, under-trained epoch.
  It finds more melanomas (129 instead of 109 of 168) but at a high price: melanoma
  specificity drops from 0.963 to 0.828 and every other summary metric gets worse.
  Validation recall also jumped between 0.49 and 0.73 from epoch to epoch, so this
  criterion is noisy.
- The weighted sampler did not change melanoma recall, but improved the rare classes
  (`df` 0.455 -> 0.591, `vasc` 0.773 -> 0.864, `bkl` 0.516 -> 0.618) and balanced accuracy.
- Neither setting improves melanoma detection without a cost. Differences of a few
  points between single runs are within what a different seed could produce.

### Decision threshold for a target sensitivity

Instead of "highest probability wins", flag an image when the probability of the
critical class (malignant / melanoma) exceeds a threshold. The threshold is chosen on
the **validation** split so that 90 % of the positives are found there, then applied
unchanged to the test split:

```bash
python -m src.evaluate --checkpoint runs/<run_name>/best.pt --target-sensitivity 0.9
```

| Run | Flagged class | Threshold | Test sensitivity | Test specificity | Test precision | Missed | False alarms |
|---|---|---|---|---|---|---|---|
| Binary | malignant | 0.1115 | 0.882 | 0.816 | 0.542 | 35 of 296 | 221 |
| Multiclass, `weighted_loss` | melanoma | 0.0001 | 0.905 | 0.759 | 0.322 | 16 of 168 | 320 |
| Multiclass, `weighted_sampler` | melanoma | 0.0023 | 0.899 | 0.830 | 0.401 | 17 of 168 | 226 |

- Compared with argmax, melanoma sensitivity rises from about 0.65 to about 0.90
  without retraining, paid for with many more false alarms (precision 0.32-0.40:
  most flagged images are not melanoma).
- At equal sensitivity the `weighted_sampler` model raises about 30 % fewer false
  alarms than the `weighted_loss` model - a difference argmax metrics did not show.
- The thresholds are tiny (0.0001, 0.0023): the models are badly calibrated and
  over-confident, so the probabilities must not be read as risk estimates.
- A threshold tuned on about 170 validation melanomas is itself uncertain.

Results on a single internal test split of one dataset say little about real-world
performance - see the next section.

## Limitations and bias

- **Skin tone**: HAM10000 was collected in Austria and Australia and contains
  overwhelmingly light skin. Expect worse, and unmeasured, performance on darker
  skin tones. The dataset has no skin-type labels, so this bias cannot even be
  quantified here.
- **Dataset bias**: Images come from two sites with specific cameras and patient
  populations. Lesions were selected because they were clinically suspicious or
  interesting, so class frequencies do not reflect the general population. Models
  can also learn shortcuts such as rulers, ink markings, hair or dark image corners.
- **Dermatoscopy only**: The model has only seen dermatoscopic images taken with
  special equipment. It is not valid for smartphone or other clinical photos.
- **Limited classes**: Only 7 diagnostic categories are covered. Any other condition
  will still be forced into one of them.
- **Label quality**: Only a little more than half of the diagnoses are confirmed by histopathology;
  the rest rely on follow-up, expert consensus or confocal microscopy.
- **Binary grouping is a simplification**: actinic keratoses (pre-cancerous) are
  counted as malignant here.
- **No clinical context**: Age, sex, body site and patient history are ignored.
- **Uncalibrated outputs**: The probabilities are model scores, not risk estimates.
- **Internal validation only**: All numbers come from a held-out part of the same
  dataset, which typically overestimates performance on new hospitals and devices.

## Next steps

- **External validation** on an independent dataset such as
  [PH2](https://www.fc.up.pt/addi/ph2%20database.html) or ISIC 2019/2020 to measure
  generalisation (check each dataset's license and overlap with HAM10000 first).
- Probability calibration (temperature scaling) and confidence intervals (bootstrap).
- Repeat runs with several seeds to separate real differences from noise.
- Compare backbones (ResNet50, ConvNeXt-Tiny) and image sizes.
- Test-time augmentation, cross-validation over lesion-level folds.

## Acknowledgements

- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection
  of multi-source dermatoscopic images of common pigmented skin lesions.
  *Sci. Data* 5, 180161 (2018).
- Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization. *ICCV* (2017).
- Tan, M. & Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks. *ICML* (2019).

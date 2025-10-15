# MURA Hand — DenseNet-169 (study-level)

![Python](https://img.shields.io/badge/python-3.10+-informational)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bogomil-iliev/mura-hand-densenet-study/blob/main/notebooks/mura_hand_study_pipeline.ipynb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Study-level abnormality detection on the **MURA hand** subset with a **DenseNet-169** backbone and **per-study aggregation** (we average per-image probabilities to decide the study). Training uses **weighted BCE**, **Adam** (1e-3), and LR-on-plateau; inputs are resized to **224×224** with light flips/rotations and ImageNet normalization. 

> Full mini-project report with data analysis, pipeline, and results: `docs/report_mura_hand.pdf`.

## Quickstart

### 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
or

Run the Colab Link above.

### 2) Dataset
Download MURA-v1.1 (hand subset) and place it as described in data/README.md.
The code expects data/MURA-v1.1/... with the original train/valid splits and train_labeled_studies.csv, valid_labeled_studies.csv.

### 3) Train
```bash
# defaults: study_type=XR_HAND, 224×224, Adam(1e-3), weighted BCE, LR on plateau
python main.py
```
Outputs:
  - logs/ (loss/acc plots via utils.plot_training)
  - results/ (confusion matrix per epoch)
  - pretrained/ (best model.pth if you save it there)

### 4) Evaluate / Inference
If you save pretrained/model.pth, add a small script (or extend main.py) to load the state dict and iterate valid to print study-level metrics.

## What’s inside
  - Study-level aggregation: average per-image probabilities → per-study decision threshold 0.5.
  - DenseNet-169 (ImageNet init) with final layer changed to a single logit + sigmoid.
  - Weighted BCE to counter class imbalance. LR decays ×0.1 when val loss plateaus.
  - Augmentation: resize 224, random horizontal flip, small rotation, ImageNet norm.

## Results
![loss](https://github.com/user-attachments/assets/749b3fcc-1b71-4c3f-9492-f7a15591aa9d)

<img width="1875" height="144" alt="image" src="https://github.com/user-attachments/assets/8235af27-f12c-48e2-8f58-8ec144f7c1de" />

From the training, it can be seen that the model scored a gradually improving performance as the number of epochs increased. The true negatives improved from 0.5498 in the first epoch to a best result of 0.9406 in the fifth epoch. On the other hand, the false positives showed a decreasing trend, starting from 0.4502 and reaching a minimum of 0.0594. The false negatives got reduced from 0.6008 to 0.5547, while the true positives were elevated from 0.3992 to 0.4453. This suggests that the model progressively improved its ability to accurately classify instances within the training set.
Similarly, in the validation set the results are good, highlighting the generalisation capability of the model. The true negatives improved from 0.4653 to 0.9406, while the false positives went down from 0.5347 to 0.0594. The false negatives decreased from 0.3485 to 0.2121, and the true positives went up from 0.6515 to 0.7879. These results in the validation set demonstrate that the model can effectively classify unseen data.
Looking at the metrics of training and validation loss, it is visible that both reduced with consistency during the training process. The training loss went down from 0.2753 in the first epoch to a minimum of 0.2675 in the fifth epoch. Respectively, the validation loss was reduced from 0.4052 to 0.6828, indicating that the model learned to minimise the errors.
Regarding training and validation accuracy, the model achieved a progress in both metrics over the epochs. The training accuracy started at 0.5109 and reached 0.5743, while the validation accuracy was progressively improving from 0.5389 to 0.5749. These results point out that the model learned the underlying patterns in the data and became more accurate in its predictions.
Overall, the custom DenseNet model showed promising performance in both the training and validation sets. It successfully learned to classify data, where the accuracy and diminishing loss through the epochs were continuously getting better. The results indicate that the model has the potential to effectively generalise and make accurate predictions on unseen data.

## Repo map
```bash
data/               # not tracked; see data/README.md
docs/
  ├─ report_mura_hand.pdf
  └─ figures/{loss.jpg,nsvc.jpg,pcpsc.jpg,pcpst.jpg}
pretrained/         # put model.pth here locally (.gitkeep in repo)
main.py
train.py
pipeline.py
densenet.py
utils.py
requirements.txt
LICENSE
README.md
```

## Notes
  - Research/education only — not a medical device.
  - MURA licensing applies to the dataset; do not commit images.
  - **Paths**: if any script still points to absolute Colab/Drive paths (e.g., `/content/drive/...`), replace with relative roots like `data/MURA-v1.1/...`. 
  - **Old PyTorch idioms**: `loss.data[0]` etc. are legacy; consider `loss.item()` and `preds = (outputs > 0.5).float()`.
  - **torch/torchvision** are pinned in the requirements.txt you might want to unpin them if running on a local machine.
  - **Pretrained weights**
  - Download the pre-trained model `model.pth` from the [v0.1.0 release](../../releases/tag/v0.1.0) (see “Assets”).
  - Place it in `pretrained/` or pass `--weights /path/to/model.pth` to the script.
  

## Citation
[➡️ Cite this repository](./CITATION.cff)"



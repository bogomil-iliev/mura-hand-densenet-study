# Data (not in repo)
# The data can be downloaded from the below link
https://www.kaggle.com/datasets/bogomililiev2308/mura-hand-xray
Expected layout (relative to repo root):

data/
└─ MURA-v1.1/
   ├─ train/
   │  └─ XR_HAND/patient*/study*_positive|negative/image*.png
   ├─ valid/
   │  └─ XR_HAND/patient*/study*_positive|negative/image*.png
   └─ train/train_labeled_studies.csv
   └─ valid/valid_labeled_studies.csv

CSV schema (as used by the code):
Path,label
MURA-v1.1/train/XR_HAND/patient00001/study1_positive,0
...


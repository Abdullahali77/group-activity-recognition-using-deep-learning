# Group Activity Recognition Using Deep Learning

PyTorch implementation of ideas from:

**A Hierarchical Deep Temporal Model for Group Activity Recognition**  
Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, Greg Mori (CVPR 2016)

Paper link: [[LINK](https://github.com/mostafa-saad/deep-activity-rec)]

## Project Overview

This repository explores **group activity recognition in volleyball** by building multiple baselines that move from simple frame-level classification to hierarchical temporal models.

The core goal is to classify a volleyball clip into one group activity label by modeling:

- Individual player actions
- Team-level spatial structure
- Temporal dynamics across frames

## What I Implemented

I implemented and organized multiple baselines under `models/`, each with a different modeling strategy:

- `b1.py`: ResNet-50 image-level baseline for 8 group classes
- `b2.py`: Reserved baseline file (currently empty)
- `b3.py`: Frozen person-level backbone + player pooling + group classifier
- `b4.py`: ResNet feature extractor + temporal LSTM baseline
- `b5.py`: Player-wise LSTM + max pooling over players
- `b6.py`: BiLSTM temporal model over pooled frame/player features
- `b7.py`: Hierarchical player-LSTM then frame-LSTM baseline
- `model_b8.py`: Two-stage hierarchical model with person-level and group-level outputs

## Baseline Summary and Accuracy

Fill these with your final reported numbers.

| Baseline | File | Main Idea | Validation/Test Accuracy |
|---|---|---|---|
| B1 | `models/b1.py` | ResNet-50 full-frame classifier | **[80%]** |
| B2 | `models/b2.py` | Reserved / not implemented | **[NA]** |
| B3 | `models/b3.py` | Person features + pooling + MLP classifier | **[76%]** |
| B4 | `models/b4.py` | CNN features + temporal LSTM | **[77%]** |
| B5 | `models/b5.py` | Player LSTM summaries + player max pool | **[81%]** |
| B6 | `models/b6.py` | Pooled features + bidirectional LSTM | **[83%]** |
| B7 | `models/b7.py` | Player LSTM -> frame LSTM hierarchy | **[86%]** |
| B8 | `models/model_b8.py` | Hierarchical person/group model with team aggregation | **[88%]** |

### Best Baseline

- Best model: **[`models/model_b8.py`]**
- Best accuracy: **[88%]**

## Dataset and Labels

- Dataset: Volleyball group activity dataset
- Group classes (8): `r_set`, `r_spike`, `r_pass`, `r_winpoint`, `l_set`, `l_spike`, `l_pass`, `l_winpoint`
- Person/action classes: **[["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]]**
- Split details: **[The dataset contains 55 videos. Each video has a folder for it with unique IDs (0, 1...54)
Train Videos: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54
Validation Videos: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
Test Videos: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47]**

Dataset source: [[DATASET_LINK](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing)]

## Results Artifacts

Current repository artifacts:

- B1 confusion matrix: `outputs/confusion_matrix_b1.png`
- B8 confusion matrix: `outputs/confusion_matrix_b8.png`

## Tech Stack

- Python
- PyTorch / torchvision
- Albumentations
- scikit-learn
- OpenCV
- Matplotlib / Seaborn

See `requirements.txt` for exact dependencies.

## Repository Structure

```text
group-activity-recognition-using-deep-learning/
|-- data/
|-- logs/
|-- models/
|   |-- b1.py
|   |-- b2.py
|   |-- b3.py
|   |-- b4.py
|   |-- b5.py
|   |-- b6.py
|   |-- b7.py
|   |-- model_b8.py
|-- outputs/
|-- scripts/
|   |-- b8-t4.ipynb
|-- utilities/
|   |-- data.py
|   |-- train.py
|-- requirements.txt
|-- README.md
```

## Notes

- This README is intentionally focused on **project information and results**.
- Usage and run instructions are intentionally minimal.

## Citation

If you use this repository, cite the original paper:

```bibtex
@inproceedings{ibrahim2016hierarchical,
	title={A Hierarchical Deep Temporal Model for Group Activity Recognition},
	author={Ibrahim, Mostafa S. and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2016}
}
```

## License

License: This project is for educational and research purposes.
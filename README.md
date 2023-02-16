[On Out-of-Distribution Detection for Audio with Deep Nearest Neighbors](https://arxiv.org/pdf/2210.15283.pdf)
---
by Zaharah Bukhsh, Aaqib Saeed @ 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

### Abstract
Out-of-distribution (OOD) detection is concerned with identifying data points that do not belong to the same distribution as the model's training data. For the safe deployment of predictive models in a real-world environment, it is critical to avoid making confident predictions on OOD inputs as it can lead to potentially dangerous consequences. However, OOD detection largely remains an under-explored area in the audio (and speech) domain. This is despite the fact that audio is a central modality for many tasks, such as speaker diarization, automatic speech recognition, and sound event detection. To address this, we propose to leverage feature-space of the model with deep k-nearest neighbors to detect OOD samples. We show that this simple and flexible method effectively detects OOD inputs across a broad category of audio (and speech) datasets. Specifically, it improves the false positive rate (FPR@TPR95) by 17% and the AUROC score by 7% than other prior techniques.

---

We provide an example of running an OOD detection with Deep-kNN. 

#### 1. Installation
Install required packages:
```
[Tensorflow](https://www.tensorflow.org/install)
[FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
```

#### 2. Extract features with a pretrained model (Optional): 
The pretrained MobileNet (YAMNet) models are available from [this](https://drive.google.com/drive/folders/1bJTq2AyXUv_Ol59ce0n5JJ-zcxTh1se5?usp=share_link) link. Download and put them under `models` directory.
```
python3 fe.py
```
Download precomputed features from [here](https://drive.google.com/drive/folders/1bJTq2AyXUv_Ol59ce0n5JJ-zcxTh1se5?usp=share_link) and put them under `features` directory. 

#### 3. Run OOD detection with Deep kNN: 
```
python3 run_ood.py
```

### Citation
```
@article{bukhsh2022out,
  title={On Out-of-Distribution Detection for Audio with Deep Nearest Neighbors},
  author={Bukhsh, Zaharah and Saeed, Aaqib},
  journal={arXiv preprint arXiv:2210.15283},
  year={2022}
}
```

# ACTION: Towards Test-Time Video Anomaly Detection via Anomaly-Aware and Temporally Consistent Adaptation

## 📌 Overview
ACTION is a test-time adaptation framework for Video Anomaly Detection (VAD). It addresses distribution shifts in real-world videos (lighting, viewpoint changes, scene dynamics) while preventing corruption from unlabeled anomalies during adaptation.

---

## 🚀 Key Idea
Traditional VAD models struggle under domain shift. ACTION introduces:

- 🔍 **Anomaly-aware confidence filtering**
- ⏱️ **Multi-scale temporal consistency**
- 🧠 **Scene-adaptive memory module**

These components allow stable adaptation without retraining or source data access.

---

## 📷 Figures

### Framework Overview
 
 ![Uploading Qualtative_TTA.jpg…]()

 
# ACTION: Test-Time Video Anomaly Detection

## 📌 Overview
This project implements **ACTION**, a test-time adaptation framework for Video Anomaly Detection (VAD).  
It handles real-world distribution shifts (lighting, viewpoint, scene changes) using anomaly-aware adaptation and temporal consistency.

---

## 📁 Project Files

This repository contains the following main modules:

- **ST_TTA.py** → Test-Time Adaptation for ShanghaiTech (ST) dataset  
- **TTA_XD.py** → Test-Time Adaptation for XD-Violence dataset  
- **UCF_TTA.py** → Test-Time Adaptation for UCF-Crime dataset  
- **cross_dataset_6_groups.py** → Cross-dataset evaluation (6 transfer settings)

 

## 🔁 Cross-Dataset Evaluation

Evaluates performance under domain shift across datasets:

| Method | UCF→ST | ST→UCF | UCF→XD | XD→UCF | ST→XD | XD→ST |
|--------|--------|--------|--------|--------|--------|--------|

Implemented in:
## 📊 Datasets
- UCF-Crime
- XD-Violence
- ShanghaiTech
- Cross-dataset evaluation
# ACTION: Test-Time Video Anomaly Detection

 

## 📁 Project Files

This repository contains the following main modules:

- **ST_TTA.py** → Test-Time Adaptation for ShanghaiTech (ST) dataset  
- **TTA_XD.py** → Test-Time Adaptation for XD-Violence dataset  
- **UCF_TTA.py** → Test-Time Adaptation for UCF-Crime dataset  
- **cross_dataset_6_groups.py** → Cross-dataset evaluation (6 transfer settings)

 

## 🔁 Cross-Dataset Evaluation

Evaluates performance under domain shift across datasets:

| Method | UCF→ST | ST→UCF | UCF→XD | XD→UCF | ST→XD | XD→ST |
|--------|--------|--------|--------|--------|--------|--------|


## 🌐 Dataset Links

- 🔗 UCF-Crime  
https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html  

- 🔗 XD-Violence  
https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20  

- 🔗 ShanghaiTech  
https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html  




---

## ⚙️ Installation
```bash
git clone https://github.com/khansobuz/ACTION.git
cd ACTION
pip install -r requirements.txt

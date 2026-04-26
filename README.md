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

 

## 📊 Datasets
- UCF-Crime
- XD-Violence
- ShanghaiTech
- Cross-dataset evaluation

 

## 📷 Figures

### Framework Overview
 
 ![Uploading Qualtative_TTA.jpg…]()


---

## ⚙️ Installation
```bash
git clone https://github.com/khansobuz/ACTION.git
cd ACTION
pip install -r requirements.txt

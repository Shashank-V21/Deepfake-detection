# 🤖 Deepfake Detection System

This project is a deep learning-based application designed to detect deepfake videos by combining a Convolutional Neural Network (CNN) for feature extraction with an SVM classifier for classification. It features a simple GUI for uploading videos and returns whether the video is real or fake.

---

## 📌 Features
- Real-time deepfake detection with video upload
- Uses **MTCNN** for face detection and **Xception** for feature extraction
- Classifies frames using a trained **SVM model**
- Majority voting across frames to determine video authenticity
- GUI built with **Tkinter** for user interaction

---

## 🛠️ Tech Stack
- Python 3
- TensorFlow / Keras
- OpenCV
- MTCNN
- scikit-learn
- Tkinter (GUI)

---

## 🗂️ Dataset
- [FaceForensics++](https://github.com/ondyari/FaceForensics)

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model (optional)**
```bash
python deepfake_detection_proj.py  # Edit the script to call train_classifier() with real & fake video lists
```

4. **Run the app**
```bash
python deepfake_detection_proj.py
```

---

## 📁 Folder Structure
```
├── deepfake_detection_proj.py      # Main GUI + logic script
├── svm_classifier.pkl              # Pre-trained SVM model
├── requirements.txt                # Python dependencies
```

---

## 🔧 Requirements
List this inside `requirements.txt`:
```
tensorflow
opencv-python
mtcnn
scikit-learn
numpy
pillow
```

---

## 📣 Future Improvements
- Add webcam support for live detection
- Display detected faces in GUI
- Add result logging and report generation

---

## 📬 Contact
**Shashank V**  
[LinkedIn](https://linkedin.com/in/shashankv21) | [GitHub](https://github.com/Shashank-V21) | shashank.v0084@gmail.com

---

## 📄 License
This project is licensed under the MIT License.

---

## 🔖 GitHub Description (for your repo):
> Deepfake detection system using MTCNN + Xception + SVM with a Tkinter GUI. Upload videos and get real/fake classification results in real-time.

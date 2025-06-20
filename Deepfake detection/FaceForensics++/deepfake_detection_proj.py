import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image
from mtcnn.mtcnn import MTCNN


# Function to extract face embeddings using pre-trained Xception model
def extract_face_embeddings(video_path):
    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    detector = MTCNN()

    # Open video
    cap = cv2.VideoCapture(video_path)
    embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)  # Ensure no negative indices
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (299, 299))  # Resize to Xception's expected input size

            # Preprocess the image for Xception model
            img = image.img_to_array(face_img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Extract embeddings
            embedding = model.predict(img)
            embeddings.append(embedding.flatten())

    cap.release()
    return np.array(embeddings)


# Function to train SVM classifier
def train_classifier(real_videos, fake_videos):
    real_embeddings = []
    fake_embeddings = []

    # Extract embeddings for real videos
    for video in real_videos:
        real_embeddings.extend(extract_face_embeddings(video))

    # Extract embeddings for fake videos
    for video in fake_videos:
        fake_embeddings.extend(extract_face_embeddings(video))

    # Prepare data for classifier
    all_embeddings = np.vstack((real_embeddings, fake_embeddings))
    labels = ['real'] * len(real_embeddings) + ['fake'] * len(fake_embeddings)

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Train SVM classifier
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(all_embeddings, encoded_labels)

    # Save the model and label encoder
    with open('svm_classifier.pkl', 'wb') as f:
        pickle.dump((svm_classifier, le), f)

    print("Training complete. Model saved.")


# Function to predict whether a video is real or fake
def predict_video(video_path):
    with open('svm_classifier.pkl', 'rb') as f:
        svm_classifier, le = pickle.load(f)

    embeddings = extract_face_embeddings(video_path)
    if len(embeddings) == 0:
        return "No faces detected in the video."

    predictions = svm_classifier.predict(embeddings)
    predicted_labels = le.inverse_transform(predictions)

    # Majority voting
    real_count = np.sum(predicted_labels == 'real')
    fake_count = np.sum(predicted_labels == 'fake')

    return 'Real Video' if real_count > fake_count else 'Fake Video'


# GUI for uploading a video and displaying the result
class DeepfakeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detection")
        self.root.geometry("700x500")
        self.root.configure(bg="#f0f0f0")

        # Add a title label
        self.title_label = tk.Label(self.root, text="Deepfake Detection System", font=("Helvetica", 24, "bold"),
                                    bg="#f0f0f0", fg="#333")
        self.title_label.pack(pady=20)

        # Add a frame for styling
        self.frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief="groove")
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Instructions Label
        self.label = tk.Label(self.frame, text="Upload a video to detect if it's real or fake", font=("Helvetica", 14),
                              bg="#ffffff", fg="#555")
        self.label.pack(pady=10)

        # Upload Button
        self.upload_button = tk.Button(self.frame, text="Upload Video", command=self.upload_video,
                                       font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5,
                                       relief="flat")
        self.upload_button.pack(pady=10)

        # Clear Output Button
        self.clear_button = tk.Button(self.frame, text="Clear Output", command=self.clear_output,
                                      font=("Helvetica", 12, "bold"), bg="#f44336", fg="white", padx=10, pady=5,
                                      relief="flat")
        self.clear_button.pack(pady=10)

        # Result Label
        self.result_label = tk.Label(self.frame, text="", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#333")
        self.result_label.pack(pady=20)

        # About Button
        self.about_button = tk.Button(self.root, text="About", command=self.show_about, font=("Helvetica", 12),
                                      bg="#2196F3", fg="white", padx=10, pady=5, relief="flat")
        self.about_button.pack(side="bottom", pady=10)

    def upload_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])

        if video_path:
            try:
                result = predict_video(video_path)
                self.result_label.config(text=f"The uploaded video is: {result}")
            except Exception as e:
                self.result_label.config(text=f"Error: {str(e)}")

    def clear_output(self):
        self.result_label.config(text="")

    def show_about(self):
        messagebox.showinfo("About",
                            "Deepfake Detection System\nVersion 1.0\n\nThis application detects whether a video is real or fake using machine learning techniques.")


# Initialize and run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectionApp(root)
    root.mainloop()

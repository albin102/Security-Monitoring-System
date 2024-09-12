import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.applications.inception_resnet_v2 import preprocess_input
from joblib import dump  # Import the correct dump function
import matplotlib.pyplot as plt
import seaborn as sns

# Load FaceNet model
facenet = FaceNet()

# Load MTCNN model for face detection
mtcnn = MTCNN()

# Function to extract FaceNet embeddings for a face
def get_face_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    embedding = facenet.model.predict(preprocess_input(face))[0]
    return embedding

# Function to detect faces in an image using MTCNN
def detect_faces(img):
    result = mtcnn.detect_faces(img)
    faces = [img[y:y+h, x:x+w] for (x, y, w, h) in [box['box'] for box in result]]
    return faces

# Function to load images from a folder and extract embeddings
def load_images_from_folder(folder):
    images = []
    labels = []
    for person_folder in os.listdir(folder):
        person_path = os.path.join(folder, person_folder)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Detect faces in the image
                    faces = detect_faces(img)
                    
                    # Get FaceNet embeddings for each detected face
                    for face in faces:
                        embedding = get_face_embedding(face)
                        images.append(embedding)
                        labels.append(person_folder)
                        cv2.imshow('Detected Face', face)
                        cv2.waitKey(100)
                        cv2.destroyAllWindows()

    return images, labels

dataset_folder = 'Dataset'  
images, labels = load_images_from_folder(dataset_folder)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))

X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Define a simple neural network
model = Sequential()
model.add(Dense(128, input_dim=512, activation='relu'))  # Update input_dim to 512
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the neural network
model.fit(np.array(X_train), y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
accuracy = model.evaluate(np.array(X_test), y_test_onehot)[1]
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Save the trained model and label encoder
model_save_path = 'Model/'
os.makedirs(model_save_path, exist_ok=True)
model.save(os.path.join(model_save_path, 'neural_network_model_new.h5'))
label_encoder_path = os.path.join(model_save_path, 'label_encoder_new.joblib')

# Use joblib.dump to save the label encoder
dump(label_encoder, label_encoder_path)

# Predict the labels for the test set
y_pred = model.predict(np.array(X_test))
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the classification report
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
print(report)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

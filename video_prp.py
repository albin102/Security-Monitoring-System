import tkinter as tk
from ultralytics import YOLO
import cv2
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from joblib import load
from keras_facenet import FaceNet
from keras.applications.inception_resnet_v2 import preprocess_input
import os

wp_model = YOLO("best.pt")

def gun(path):
		video_capture = cv2.VideoCapture(path) 
		while True:
			ret, frame = video_capture.read()
			
			if not ret:
				break
			frame = cv2.resize(frame, (800, 600))
			weapon_detection(frame)
		
		video_capture.release()
		cv2.destroyAllWindows()

def weapon_detection(frame):
    results_w = wp_model(frame, show=True)
    print(results_w)
    confidence_w = np.array(results_w[0].boxes.conf)

def detect_fr(pathh,dire):
	facenet = FaceNet()
	frames_folder = dire
	# Load the trained model and label encoder
	model_save_path = 'Suspicious Detection/Model/'
	model = load_model(os.path.join(model_save_path, 'neural_network_model12.h5'))
	# label_encoder = load(os.path.join(model_save_path, 'label_encoder6.joblib'))

	# Load face detection model
	face_detector = MTCNN()

	class_idx_to_celebrity = {0: 'criminal1', 1: 'crminal2'}

	# Normalize embeddings
	# l2_normalizer = Normalizer('l2')

	# Open video capture
	video_capture = cv2.VideoCapture(pathh)  # Use the provided video_file parameter

	# Get the original frame dimensions
	frame_width = int(video_capture.get(3))
	frame_height = int(video_capture.get(4))

	# Define the resized dimensions
	resized_width = 800
	resized_height = 600

	# Define the skip frames parameter
	skip_frames = 8  # Process every 5th frame

	# Initialize frame count
	frame_count = 0

	# Set to store detected celebrities
	detected_celebrities = set()

	while True:
		# Read a frame from the video stream
		ret, frame = video_capture.read()
		if not ret:
			break

		# Skip frames
		if frame_count % skip_frames == 0:
			# Resize the frame
			frame = cv2.resize(frame, (resized_width, resized_height))

			# Convert the frame to RGB
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Detect faces in the frame
			faces = face_detector.detect_faces(rgb_frame)

			# Loop through detected faces
			for face in faces:
				x, y, w, h = face['box']
				face_img = frame[y:y+h, x:x+w]
				face_img = cv2.resize(face_img, (160, 160))
				face_img = np.expand_dims(face_img, axis=0)
				embedding = facenet.model.predict(preprocess_input(face_img))[0]

				# Make predictions using the trained model
				predictions = model.predict(np.array([embedding]))
				predicted_class_idx = np.argmax(predictions)
				confidence = predictions[0][predicted_class_idx]

				# Threshold for considering a prediction as correct
				threshold = 0.95

				if confidence > threshold:
					# Known celebrity
					celebrity_name = class_idx_to_celebrity.get(predicted_class_idx, 'Unknown')
					detected_celebrities.add(celebrity_name)

					color = (0, 255, 0)  # Green for known celebrities
					text = f"{celebrity_name} ({confidence:.2f})"
					frame_filename = f"{celebrity_name}_{frame_count}.jpg"
					frame_filepath = os.path.join(frames_folder, frame_filename)
					cv2.imwrite(frame_filepath, frame)
				else:
					# Unknown person
					color = (0, 0, 255)  # Red for unknown persons
					text = "Unknown"

				# Draw bounding box and label on the frame
				cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
				cv2.putText(frame, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			# Display the resulting frame
			cv2.imshow('Video', frame)

			# Break the loop if 'q' key is pressed
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		frame_count += 1

	# Release video capture and close all windows
	video_capture.release()
	cv2.destroyAllWindows()
	x=list(detected_celebrities)

	l= len(x)
	k=[]
	print(l)
	for i in range  (0,l):
		if str(x[i])=="criminal1":
			name = "Irrfan Khan"
		elif str(x[i]) == "criminal2":
			name = "Ranbir Kapoor"
		else:
			name = "unknwon"
		k.append(name)

	

def video_check(path):
	filename = os.path.basename(path)
	filename_with_extension = os.path.basename(path)
	filename_without_extension = os.path.splitext(filename_with_extension)[0]
	print(filename_without_extension)
	directory="Frames/Face/"+filename_without_extension
	if not os.path.exists(directory):
		os.makedirs(directory)
	detect_fr(path,directory)
	gun(path)


	
	

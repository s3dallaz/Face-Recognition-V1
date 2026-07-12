import face_recognition
import numpy as np
import cv2 as cv

from load_database import load_database
from face_detector import face_detector
from face_encoder import face_encoder
from config import EMBEDDINGS_PATH




def recognize_face(current_encoding, known_encodings, known_names , threshold = 0.6):

    distances = face_recognition.face_distance(known_encodings , current_encoding) 

    best_match = np.argmin(distances)   # Smallest arg index
    best_distance = distances[best_match]


    if best_distance > threshold:

        return "unknown"

    best_name = known_names[best_match]

    return best_name



# recognize_face(test_encode[0], known_encodings, known_names)

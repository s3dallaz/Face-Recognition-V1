import os 
import cv2 as cv
import numpy as np

from face_detector import face_detector
from face_encoder import face_encoder
from config import DATASET_PATH


def build_database(dataset_path , embeddings_path , person_name= None):


    new_encodings = []
    new_names = []

    if person_name is None:
        ds_content = os.listdir(dataset_path)

    else:
        ds_content = [person_name]

    for folder_name in ds_content:

        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        imgs_names = os.listdir(folder_path)

            
        for img_name in imgs_names:

            imgs_paths = os.path.join(folder_path , img_name)

            img = cv.imread(imgs_paths)

            face_locations = face_detector(img)

            if len(face_locations) != 1:
                continue

            
            encoding = face_encoder(img , face_locations)

            person_database_path = os.path.join(embeddings_path , folder_name)

            name , ext = os.path.splitext(img_name)
            new_name = f"{name}.npy"

            os.makedirs(person_database_path , exist_ok= True)


            img_encodes_paths = os.path.join(person_database_path , new_name)

            np.save(img_encodes_paths , encoding[0])

            new_encodings.append(encoding[0])
            new_names.append(folder_name)

            
    return new_encodings, new_names



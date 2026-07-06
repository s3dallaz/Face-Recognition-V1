import os 
import cv2 as cv
import numpy as np

from face_detector import face_detector
from face_encoder import face_encoder
from config import EMBEDDINGS_PATH


dataset_path = (r"E:\Projects\Face-Recognition-V1\dataset")
ds_content = os.listdir(dataset_path)

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

        person_database_path = f"E:\Projects\Face-Recognition-V1\src\database\{folder_name}"

        name , ext = os.path.splitext(img_name)
        new_name = f"{name}.npy"

        img_encodes_paths = os.path.join(person_database_path , new_name)

        np.save(EMBEDDINGS_PATH , encoding[0])
        print(img_encodes_paths)

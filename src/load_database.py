import os 
import numpy as np

from config import EMBEDDINGS_PATH




def load_database(database_path):


    everyone_encode_paths = os.listdir(database_path)    # person folders    E:\Projects\Face-Recognition-V1\database\

    known_encodings = []
    known_names = []

    for person_folder in everyone_encode_paths:     # person specific folder

        folder_path = os.path.join(database_path , person_folder)     # E:\Projects\Face-Recognition-V1\database\abdelrahman

        if not os.path.isdir(folder_path):
            continue

        person_encode_path = os.listdir(folder_path)     # open file names path   ["img1.npy","img2.npy"]
        

        for encode_file in person_encode_path:

            file_path = os.path.join(folder_path , encode_file)    # E:\Projects\Face-Recognition-V1\database\abdelrahman\img1.npy

            encode = np.load(file_path)

            known_encodings.append(encode)
            known_names.append(person_folder)


    return known_encodings , known_names


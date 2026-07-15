import cv2 as cv
import os
from config import DATASET_PATH
from face_detector import face_detector
from build_database import build_database
from config import EMBEDDINGS_PATH
from load_database import load_database




def capture_person_images(cap , person_name , img_num= 20):
    SPACE_KEY = 32
    ESC_KEY = 27
    
    img_count = 0

    new_person_path = os.path.join(DATASET_PATH , person_name)

    if os.path.exists(new_person_path):
        print(f"{person_name} already exists.")
        return False

    os.makedirs(new_person_path , exist_ok= True)

    while True:
        success , frame = cap.read()
    
        if not success:
            return False
        
        cv.imshow("Frame", frame)

        key = cv.waitKey(1)

        if key == ESC_KEY:
            return False
        


        elif key == SPACE_KEY:
            
            face_locations = face_detector(frame)

            if len(face_locations) != 1:
                print("Exactly one face must be visible.")
                continue

            file_name = f"img{img_count + 1}.jpg" 
            img_count += 1

            img_path = os.path.join(new_person_path , file_name)

            cv.imwrite(img_path , frame)

            print(f"Saved image {img_count}/{img_num}")

        if img_count >= img_num:
            print(f"{person_name}'s image capture completed.\n\n")
            print("Processing images... \n")
            print("Please wait, That should take only few seconds...\n")
            print("NOTE! : Please don't press any buttons until the camera start again!\n")
            return True
    

    




def add_new_person(cap , person_name):

    person_name = person_name.strip()

    if not person_name:
        print("Name cannot be empty.")
        return [], []

    success = capture_person_images(cap, person_name)

    if not success:
        return [], []
        

    show_controls()

    new_encodings, new_names = build_database(DATASET_PATH, EMBEDDINGS_PATH, person_name)


    return new_encodings, new_names





def show_controls():
    print("\nAdd new member: (Press A)")
    print("Exit: (Press ESC)\n")
    


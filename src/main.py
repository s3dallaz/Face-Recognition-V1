import cv2 as cv

from load_database import load_database
from face_detector import face_detector
from face_encoder import face_encoder
from recognizer import recognize_face
from visualization import draw_rectangle
from add_new_person import add_new_person , show_controls
from config import EMBEDDINGS_PATH


known_encodings , known_names = load_database(EMBEDDINGS_PATH)

RED = (0,0,255)
GREEN = (0,255,0)
ESC_KEY = 27

show_controls()

cap = cv.VideoCapture(0)

while True:
    

    success, frame = cap.read()

    if not success:
        break

    

    face_locations = face_detector(frame)

    encodings = face_encoder(frame, face_locations)
    
    for encode , location in zip(encodings , face_locations):

        top, right, bottom, left = location

        name , distance = recognize_face(encode , known_encodings , known_names , threshold= 0.6)

        distance = f"{distance:.2f}"
    


        if name == "Unknown":
            color = RED
            
        else:
            color = GREEN

        frame = draw_rectangle(frame, location, color)

        cv.putText(
            frame,
            name,
            (left, max(20, top - 10)),
            fontFace= cv.FONT_HERSHEY_SIMPLEX,
            fontScale= 0.7,
            color= color,
            thickness= 2
            )
        
        cv.putText(
            frame,
            (f"Distance: {distance}"),
            (left, max(50, bottom + 25)),
            fontFace= cv.FONT_HERSHEY_SIMPLEX,
            fontScale= 0.7,
            color= (0,255,255),
            thickness= 2
            )


    cv.imshow("Frame", frame)  

    key = cv.waitKey(1)

  
    if key == ESC_KEY:
        break


    if key == ord("a"):

        new_name = input("Please enter a name!")

        new_encodings, new_names = add_new_person(cap, new_name)

        known_encodings.extend(new_encodings)
        known_names.extend(new_names)

        






    


cap.release()
cv.destroyAllWindows()

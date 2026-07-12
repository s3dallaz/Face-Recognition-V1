import cv2 as cv

from load_database import load_database
from face_detector import face_detector
from face_encoder import face_encoder
from recognizer import recognize_face
from visualization import draw_rectangle
from config import EMBEDDINGS_PATH


known_encodings , known_names = load_database(EMBEDDINGS_PATH)

RED = (0,0,255)
GREEN = (0,255,0)


cap = cv.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    face_locations = face_detector(frame)

    encodings = face_encoder(frame, face_locations)
    

    for encode , location in zip(encodings , face_locations):

        top, right, bottom, left = location

        name = recognize_face(encode , known_encodings , known_names , threshold= 0.6)


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


    cv.imshow("Frame", frame)  
  
    if cv.waitKey(1) == 27:
        break


cap.release()
cv.destroyAllWindows()

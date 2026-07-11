import cv2 as cv

from load_database import load_database
from face_detector import face_detector
from face_encoder import face_encoder
from recognizer import recognize_face
from visualization import draw_rectangle
from config import EMBEDDINGS_PATH

print("Main started")

known_encodings , known_names = load_database(EMBEDDINGS_PATH)


cap = cv.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    face_locations = face_detector(frame)

    encodings = face_encoder(frame, face_locations)

    # Recognition

    location_connector = zip(encodings , face_locations)
    
    frame = draw_rectangle(frame, face_locations)

    for encode , location in location_connector:

        top, right, bottom, left = location

        name = recognize_face(encode , known_encodings , known_names , threshold= 0.6)

        cv.putText(
            frame,
            name,
            (left, top-30),
            fontFace= cv.FONT_HERSHEY_SIMPLEX,
            fontScale= 0.7,
            color= (0,255,0),
            thickness= 2
            )


    cv.imshow("Frame", frame)  
  
    if cv.waitKey(1) == 27:
        break


cap.release()
cv.destroyAllWindows()

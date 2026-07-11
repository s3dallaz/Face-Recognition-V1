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

    for encode in encodings:

        name = recognize_face(encode , known_encodings , known_names , threshold= 0.6)

        print(name) 

        frame = draw_rectangle(frame, face_locations)

        cv.imshow("Frame", frame)


    if cv.waitKey(1) == 27:
        break

    



cap.release()
cv.destroyAllWindows()

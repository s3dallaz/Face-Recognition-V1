import face_recognition


def face_detector(frame):

    face_locations = face_recognition.face_locations(frame)

    return face_locations

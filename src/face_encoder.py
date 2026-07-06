import face_recognition


def face_encoder(frame , face_locations):

    encodings = face_recognition.face_encodings(frame , face_locations)

    return encodings
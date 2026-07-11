import cv2 as cv


def draw_rectangle(frame , face_locations):

    for location in face_locations:
        top , right , bottom , left = location

        top_left = (left , top  )
        bottom_right = ( right , bottom )

        cv.rectangle(frame , top_left , bottom_right , color = (0,255,0),  thickness=4)

    return frame



def face_crop(frame , face_locations):    # Learning purposes only (*For future upgrades*)

    faces = []

    for location in face_locations:

        top, right, bottom, left = location

        face = frame[top : bottom , left : right]

        faces.append(face)

    return faces



# frame = capture_webcam(0)    #  Webcam capture 

# face_locations = face_detector(frame)   #  Location of faces as []

# encodes = face_encoder(frame , face_locations)  # Faces encodes

# print(type(encodes))
# print(len(encodes))
# print(type(encodes[0]))
# print(encodes[0].shape)

# # cv.imshow("Detection", draw_rectangle(frame , face_locations) )

# # cv.waitKey(0)
# # cv.destroyAllWindows()
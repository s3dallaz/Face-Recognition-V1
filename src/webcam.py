import cv2 as cv

# if __name__ == "__main__":
    
def capture_webcam(webcam_num):

    cap = cv.VideoCapture(webcam_num)             # Capture device


    while True:
        success , frame = cap.read()     # Read frame

        if not success:
            print("Failed to read frame from the camera.")
            break
            
            
        cv.imshow("cam", frame)          # Show capture
        


        ESC_KEY = 27
        
        if cv.waitKey(1) == ESC_KEY:          # Close capture (Press ESC)
            break
            


    cap.release()
    cv.destroyAllWindows()

    return frame



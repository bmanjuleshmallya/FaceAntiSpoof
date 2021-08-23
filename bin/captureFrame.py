import cv2
import sys
import dlib
from pandas.core import frame

# faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
hog_detector = dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    # ret, frame = video_capture.read()
    frame = cv2.imread('dataset/test/0/344.jpg')
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = hog_detector(frame, 1)
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # Draw a rectangle around the faces

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for i, d in enumerate(dets):
        rect = cv2.rectangle(frame, (d.bottom(), d.top()), (d.left(), d.right()), (255, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'name',(d.bottom(),d.top()), font, 1, (255,0,0)) #---write the text

        # crop = img[d.top():d.bottom(), d.left():d.right()]
        # if crop.size==0:
        #     print("Ignored this - crop picture seems to be empty")
        # else:
        #     cv2.imwrite(newPath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
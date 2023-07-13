import os
import cv2

video_path = os.path.join(os.getcwd(), 'assets', 'people.mp4')

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
while ret:
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
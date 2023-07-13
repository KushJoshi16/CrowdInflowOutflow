import os
import random

import cv2

from ultralytics import YOLO

from tracker import Tracker

video_path = os.path.join(os.getcwd(), 'assets', 'people.mp4')
video_out_path = os.path.join(os.getcwd(), 'assets', 'people_out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

model = YOLO('yolov8n.pt')

tracker = Tracker()

colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(100)]
while ret:

    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            score = float(score)
            class_id = int(class_id)

            if score > 0.5:
                detections.append([x1, y1, x2, y2, score, class_id])


    tracker.update(frame, detections)
    
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        # class_id = track.class_id
        # score = track.score

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id%len(colors)], 2)
        cv2.putText(frame, f'{track_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[track_id%len(colors)], 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
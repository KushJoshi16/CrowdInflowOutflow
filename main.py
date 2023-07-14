import os
import random

import cv2

from ultralytics import YOLO

from tracker import Tracker
import numpy as np
import copy

from frame import Frame



video_path = os.path.join(os.getcwd(), 'assets', 'people.mp4')
video_out_path = os.path.join(os.getcwd(), 'assets', 'people_out.mp4')

cap = cv2.VideoCapture(video_path)

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1_area = int(0.10 * video_width)
y1_area = int(0.10 * video_height)
x2_area = int(0.90 * video_width)
y2_area = int(0.90 * video_height)

ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

model = YOLO('yolov8n.pt')

tracker = Tracker()
cur_frame = Frame((x1_area, y1_area, x2_area, y2_area))
n_init = 3


colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(70)]
while ret:
    count = np.array([0,0,0,0])
    # count ( out, in, inflow, outflow)

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

            if score > 0.35 and class_id == 0:
                detections.append([x1, y1, x2, y2, score, class_id])


    tracker.update(frame, detections)
    if n_init > 0:
        n_init -= 1
    else:
        cur_frame.update(tracker.tracks)
        for track in tracker.tracks:
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id%len(colors)], 2)
            cv2.putText(frame, f'{track_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[track_id%len(colors)], 1)
            track_state = cur_frame.tracks_state[track_id]
            count[track_state[0]] += 1
            cv2.putText(frame, f'{cur_frame.state_class[track_state[0]]}', (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[track_id%len(colors)], 1)
        
        cv2.putText(frame, f'{count[1]},{str(" ") if count[2]==0 else count[2]},{count[3]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.rectangle(frame, (x1_area, y1_area), (x2_area, y2_area), (0, 0xFF, 0), 2)
    cv2.imshow('frame', frame)

    cv2.waitKey(1)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
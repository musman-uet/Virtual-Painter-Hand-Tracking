import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ========== MEDIAPIPE SETUP ==========
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5, 
    min_hand_presence_confidence=0.2, 
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

brush_thickness = 15
eraser_thickness = 80
draw_color = (0, 255, 0) 
xp, yp = 0, 0 
canvas = np.zeros((720, 1280, 3), np.uint8)
smoth_val = 5 

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    frame_timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # --- Colour Options at the top ---
    cv2.rectangle(img, (0, 0), (320, 100), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(img, (320, 0), (640, 100), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(img, (640, 0), (960, 100), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (960, 0), (1280, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "ERASER", (1050, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    # --- Slider ---
    cv2.rectangle(img, (1240, 150), (1260, 350), (200, 200, 200), cv2.FILLED) 
    cv2.putText(img, "SIZE", (1210, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            itip = hand_landmarks[8]
            ijoint = hand_landmarks[6]
            
            target_x, target_y = int(itip.x * 1280), int(itip.y * 720)
            
            if xp == 0 and yp == 0:
                x1, y1 = target_x, target_y
            else:
                x1 = xp + (target_x - xp) // smoth_val
                y1 = yp + (target_y - yp) // smoth_val

            index_up = itip.y < ijoint.y
            middle_up = hand_landmarks[12].y < hand_landmarks[10].y

            # Selection Mode (two fingers up)
            if index_up and middle_up:
                xp, yp = 0, 0  # Reset drawing position
                
                # Color selection in toolbar area
                if y1 < 100:
                    if 0 < x1 < 320:
                        draw_color = (0, 0, 255)
                    elif 320 < x1 < 640:
                        draw_color = (255, 0, 0)
                    elif 640 < x1 < 960:
                        draw_color = (0, 255, 0)
                    elif 960 < x1 < 1280:
                        draw_color = (0, 0, 0)
                
                # Brush size slider
                if x1 > 1200 and 150 < y1 < 350:
                    brush_thickness = int(np.interp(y1, [150, 350], [40, 5]))
                    cv2.rectangle(img, (1240, y1), (1260, 350), draw_color, cv2.FILLED)

                cv2.circle(img, (x1, y1), brush_thickness, draw_color, cv2.FILLED)

            # Drawing Mode (only index finger up)
            elif index_up and not middle_up:
                
                # Don't draw in toolbar area
                if y1 < 120:
                    xp, yp = 0, 0  # Reset so no line connects
                    cv2.circle(img, (x1, y1), brush_thickness, draw_color, cv2.FILLED)
                else:
                    # Normal drawing below toolbar
                    cv2.circle(img, (x1, y1), brush_thickness, draw_color, cv2.FILLED)
                    if xp == 0 and yp == 0: 
                        xp, yp = x1, y1

                    thickness = eraser_thickness if draw_color == (0,0,0) else brush_thickness
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                    xp, yp = x1, y1

    # Visual Preview at bottom right
    cv2.circle(img, (1250, 650), brush_thickness, draw_color, cv2.FILLED)

    # Merge canvas with camera feed
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Portfolio Project: Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
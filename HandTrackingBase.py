import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1,
                      0.65, 0.65)
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    # convert, since mp.Hands only supports RGB images
    imgRBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # processing the frame
    results = hands.process(imgRBG)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # extract finger landmarks (x, y, z coord values)
            for id, landM in enumerate(handLms.landmark):
                # print(id, landM)
                # get width and height in pixels
                h, w, channels = img.shape
                centerX, centerY = int(landM.x * w), int(landM.y * h)
                print(id, centerX, centerY)
                if id == 4:
                    cv.circle(img, (centerX, centerY), 15,
                              (255, 0, 255), cv.FILLED)

            # draw points and connections on hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    # display frame rate
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN,
               3, (255, 0, 255), 3)

    cv.imshow("Image", img)
    cv.waitKey(1)

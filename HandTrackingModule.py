import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelC = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=self.modelC,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        """
        find hands from frames
        :param img: image to find hands on
        :return: image after drawing on
        """
        # convert, since mp.Hands only supports RGB images
        imgRBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # processing the frame
        self.results = self.hands.process(imgRBG)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # draw points and connections on hand
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):

        landmark_list = []
        if self.results.multi_hand_landmarks:
            # select first hand
            myHand = self.results.multi_hand_landmarks[handNum]

            # extract finger landmarks for the hand (x, y, z coord values)
            for id, landM in enumerate(myHand.landmark):
                # print(id, landM)
                # get width and height in pixels
                h, w, channels = img.shape
                centerX, centerY = int(landM.x * w), int(landM.y * h)
                landmark_list.append([id, centerX, centerY])
                if draw:
                    cv.circle(img, (centerX, centerY), 8, (255, 0, 255), cv.FILLED)

        return landmark_list


def main():
    prevTime = 0
    currTime = 0

    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)
        if len(landmark_list) != 0:
            # choose hand landmark index
            print(landmark_list[4])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # display frame rate
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN,
                   3, (255, 0, 255), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()


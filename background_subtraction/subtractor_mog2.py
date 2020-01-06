import cv2
import numpy as np
import os

images = '../highway/input/in%06d.jpg'
res_path_frm = 'results/frame_diff/'

cap = cv2.VideoCapture(images)

#cap = cv2.VideoCapture("highway.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

i = 1
while True:
    _, frame = cap.read()

    if np.shape(frame) == ():
        break

    mask = subtractor.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)

    # Outputs to the result folder
    cv2.imwrite(os.path.join(res_path_frm, 'out%06d.jpg' % i), difference)
    i += 1

    key = cv2.waitKey(10)
    if key == 7:
        break

cap.release()
cv2.destroyAllWindows()

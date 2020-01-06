import cv2
import os

import numpy as np
from PythonCode2012.PythonCode.pythonc.Stats import Stats
from MethodsRanker.MethodsRanker import main


images = './highway/input/in%06d.jpg'
res_path_frm = 'results/frame_diff/'
#sts = Stats(path=images)
#res = sts.writeOverallResults()
#print(res)
cap = cv2.VideoCapture(images)

main()

#cap = cv2.VideoCapture("highway.mp4")
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
i = 1;
while True:
    _, frame = cap.read()

    if np.shape(frame) == ():
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("First frame", first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("difference", difference)

    # Outputs to the result folder
    cv2.imwrite(os.path.join(res_path_frm, 'out%06d.jpg' % i), difference)
    i += 1

    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
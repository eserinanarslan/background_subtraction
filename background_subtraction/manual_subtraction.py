import cv2
import os

import numpy as np
from background_subtraction.subtractor_mog2 import subtractor_mog
from PythonCode2012.PythonCode.pythonc.Stats import Stats

def evaluate(category, last, res_path):
    acc = []
    gt_path = category + '/groundtruth/'
    #start from first ground truth available
    if category == 'highway':
        i = 470
    elif category == 'pets':
        i = 570
    else:
        i = 300

    while i < last:
        #pr: prediction vs gt: ground truth
        prediction = cv2.imread(os.path.join(res_path,'out%06d.jpg' %i))
        ground_truth = cv2.imread(os.path.join(gt_path,'gt%06d.png' %i))
        c = (np.sum(abs(prediction - ground_truth)) / np.sum(ground_truth))

        acc.append(1-c)
        i+=1

    #remove nans and infs from the accuracy list
    acc = [x for x in acc if ~np.isnan(x) and ~np.isinf(x)]
    print('Accuracy:', np.mean(acc))


def calc_frame_diff(cat, res_path_frm, images):
    cap = cv2.VideoCapture(images)

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
        cv2.imshow("Difference", difference)

        # Outputs to the result folder
        cv2.imwrite(os.path.join(res_path_frm, 'out%06d.jpg' % i), difference)
        i += 1

        key = cv2.waitKey(10)
        if key == 4:
            break
    cap.release()
    cv2.destroyAllWindows()
    evaluate(cat, i, res_path_frm)


categories = ['highway', 'pets']

for cat in categories:
    images = './' + cat + '/input/in%06d.jpg'
    res_path_frm = './' + cat + '/results/frame_diff/'
    res_path_median = './' + cat + '/results/median_filter/'

    print('Frame differencing for ' + cat)
    calc_frame_diff(cat, res_path_frm, images)
    print('Median filtering for ' + cat)
    subtractor_mog(cat, res_path_median, images)


Stats.writeCategoryResult(res_path_frm, cat)
Stats.writeOverallResults(res_path_frm, cat)

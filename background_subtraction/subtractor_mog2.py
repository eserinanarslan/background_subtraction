import cv2
import numpy as np
import os

def eval(category,last,res_path):
    acc = []
    gt_path = category + '/groundtruth/'
    #start from first ground truth available
    if category == 'highway':
        i = 470
    elif category == 'office':
        i = 570
    else:
        i = 300

    while i < last:
        #pr: prediction vs gt: ground truth
        pr_1 = cv2.imread(os.path.join(res_path,'out%06d.jpg' %i))
        gt_1 = cv2.imread(os.path.join(gt_path,'gt%06d.png' %i))
        c = (np.sum(abs(pr_1 -  gt_1)) / np.sum(gt_1))

        acc.append(1-c)
        i+=1

    #remove nans and infs from the accuracy list
    acc = [x for x in acc if ~np.isnan(x) and ~np.isinf(x)]
    print('Accuracy:', np.mean(acc))


def subtractor_mog(cat, images, res_path_median):
    cap = cv2.VideoCapture(images)

    _, first_frame = cap.read()

    subtractor = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=40, detectShadows=True)

    i = 1
    while True:
        _, frame = cap.read()

        if np.shape(frame) == ():
            break

        mask = subtractor.apply(frame)

        cv2.imshow("Frame", frame)
        cv2.imshow("mask", mask)

        # Outputs to the result folder
        cv2.imwrite(os.path.join(res_path_median, 'median_out%06d.jpg' % i), mask)
        i += 1

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    eval(cat, i, res_path_median)



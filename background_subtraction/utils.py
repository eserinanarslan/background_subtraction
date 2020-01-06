import cv2
import numpy as np
import os

def evaluate_accuracy(category,last,res_path):
    print("Last :", last)
    last = last+1
    acc = []
    gt_path = category + '/groundtruth/'
    #start from first ground truth available
    if category == 'pets':
        i = 670
    elif category == 'highway':
        i = 470
    else:
        i = 250

    while i < last:
        #pr: prediction vs gt: ground truth
        pr_1 = cv2.imread(os.path.join(res_path,'out%06d.jpg' %i))
        gt_1 = cv2.imread(os.path.join(gt_path,'gt%06d.png' %i))
        c = np.sum(abs(pr_1 - gt_1)) / np.sum(abs(gt_1))
        d = [d for d in c if ~np.isnan(c) and ~np.isinf(c)]
        acc.append(1-d)
        i+=1

    #remove nans and infs from the accuracy list
    #acc = [x for x in acc if ~np.isnan(x) and ~np.isinf(x)]
    print('Accuracy:', np.mean(acc))

def read_images(category):
    images = category + '/input/in%06d.jpg'
    cap = cv2.VideoCapture(images)
    return cap

def frame_diff(category):
    cap = read_images(category)
    res_path_frm = './highway/results/frame_diff/'
    _, first_frame = cap.read(0)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
    i=1 #counter for the output images
    cap = read_images(category) #To start from the beginning
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

        #Outputs to the result folder
        cv2.imwrite(os.path.join(res_path_frm,'out%06d.jpg' %i), difference)
        i+=1

        key = cv2.waitKey(10)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    eval(category,i,res_path_frm)


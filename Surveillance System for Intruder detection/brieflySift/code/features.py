#pip3 freeze > requirements.txt  # Python3

import cv2
import numpy as np
import argparse
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import time
start = time.time()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-kp", type=str, help="key points detector")
    parser.add_argument('-des', type=str, help="descriptor")
    parser.add_argument("-trans", type = str, help="transformation on image")
    parser.add_argument("-nm", default=50, type=int, help="Number of matches")
    args = parser.parse_args()
    return args

def read_image(path1, path2):
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2

def convert_to_grayscale(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray scaled Source', gray1)
    #cv2.imshow('Gray scaled Destination', gray2)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return (gray1,gray2)

def FAST_KP(image1, image2):
    fast = cv2.FastFeatureDetector_create(threshold= threshold)
    kp1 = fast.detect(image1, None)
    kp2 = fast.detect(image2, None)
    #print(len(kp1))
    print(len(kp2))
    return kp1, kp2

def DOG_KP(image1, image2):
    sift = cv2.SIFT_create(nf)
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img2, None)
    return kp1, kp2

def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) #Or L1
    no_of_matches = brute_force.match(des1, des2)
    no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
    return no_of_matches

def display_output(pic1, kpt1, pic2, kpt2, best_match):
    match_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)
    plt.figure(figsize=(15, 5))
    plt.imshow(match_image)
    plt.show()
    #cv2.imshow('kp_des_trans_nPoints image', match_image)
    #cv2.imwrite("incorrect.png", match_image)

if __name__ == "__main__":
    args = get_args()

    path1 = f"data/{str(args.trans)}_S.ppm"
    path2 = f"data/{str(args.trans)}_T.ppm"

    img1, img2 = read_image(path1, path2)
    gray1, gray2 = convert_to_grayscale(img1, img2)

    kp = args.kp
    des = args.des
    nm = args.nm

    def x(kp, des, nm) :
        if (str(kp) == "FAST") :
            kp1, kp2 = FAST_KP(gray1, gray2)
        elif (str(kp) == "DOG") :
            kp1, kp2 = DOG_KP(gray1, gray2)

        if (str(des) == 'SIFT') :
            sift = cv2.SIFT_create(300)
            kp1, des1 = sift.compute(gray1, kp1)
            kp2, des2 = sift.compute(gray2, kp2)
        elif (str(des) == 'BREIF') :
            brief = cv2.BriefDescriptorExtractor_create(300)
            kp1, des1 = brief.compute(gray1, kp1)
            kp2, des2 = brief.compute(gray2, kp2)

        number_of_matches = BF_FeatureMatcher(des1, des2)
        print(f'Total Number of Features matches found are {len(number_of_matches)}')
        display_output(gray1, kp1, gray2, kp2, number_of_matches[0:nm])
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    for i in range(0, 2):
        if i == 0:
            print("Iteration 1")
            nf = 300
            threshold = 81  # For light = 87, rot = 127, scale = 59, view = 81
            x(kp, des, nm)
        else:
            print("Iteration 2")
            nf = 1000
            threshold = 39  # For light = 59, rot = 93, scale = 32, view = 39
            x(kp, des, nm)
    end =time.time()
    print(f"Runtime of the program is {end - start}")
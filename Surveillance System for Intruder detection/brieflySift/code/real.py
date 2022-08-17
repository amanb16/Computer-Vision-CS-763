#pip3 freeze > requirements.txt  # Python3

import cv2
from matplotlib import image
import numpy as np
import argparse
import tkinter
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--kp", type=str, help="key points detector")
    # parser.add_argument('--des', type=str, help="descriptor")
    parser.add_argument("--trans", type = str, help="transformation on image")
    parser.add_argument("--nm", type=int, help="Number of matches")
    args = parser.parse_args()
    return args

def read_image(path1, path2):
    img1 = cv2.imread(path1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(path2)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # print(img1)
    # img1 = img1.astype('uint8')
    # img2 = img2.astype('uint8')
    return img1, img2

def convert_to_grayscale(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return (gray1,gray2)

def STAR_detector(image1, image2):
    star = cv2.xfeatures2d.FREAK_create(1000)
    kp1 = star.detect(image1, None)
    kp2 = star.detect(image2, None)
    return kp1, kp2


def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) #Or L1
    no_of_matches = brute_force.match(des1, des2)
    no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
    return no_of_matches

def display_output(pic1, kpt1, pic2, kpt2, best_match):
    match_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)
    # cv2.namedWindow("Matching", cv2.WINDOW_NORMAL)
    # cv2.imshow('Matching', match_image)
    plt.imshow(match_image)
    plt.savefig('../results1.png')

if __name__ == "__main__":
    args = get_args()

    path1 = f"../data/real_and_syn_images/real_{args.trans}_S.jpg"
    path2 = f"../data/real_and_syn_images/real_{args.trans}_T.jpg"
    img1, img2 = read_image(path1, path2)
    gray1, gray2 = convert_to_grayscale(img1, img2)

    # kp = args.kp
    # kp1, kp2 = STAR_detector(gray1, gray2)

    surf = cv2.xfeatures2d.SURF_create(600)
    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)


    nm = args.nm
    number_of_matches = BF_FeatureMatcher(des1, des2)
    print(f'Total Number of Features matches found are {len(number_of_matches)}')
    print(number_of_matches[0].imgIdx, len(kp1), len(kp2))
    display_output(img1, kp1, img2, kp2, number_of_matches[:nm])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
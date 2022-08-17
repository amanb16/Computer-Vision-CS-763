import cv2 
import numpy as np
import argparse
import matplotlib.pyplot as plt

def alpha(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    row, col, ch= img1.shape

    # FOR PERPECTIVE TRANSFORM
    pts1 = np.array([[1512, 169], [2954, 726], [2998, 2050], [1493, 2221]]).astype(np.float32)
    pts2 = np.array([[1329, 340], [3015, 625], [3030, 1893], [1314, 2018]]).astype(np.float32)

    
    # pts1, pts2= haha.get_points(img1, img2)
    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    # result = cv2.warpPerspective(img2, matrix, (col, row))

    # return result
    return matrix

def beta(img2_path, img3_path):
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)

    row, col, ch= img2.shape

    #FOR PERSPECTIVE TRASNFORM
    pts1 = np.array([[1329, 340], [3015, 625], [3030, 1893], [1314, 2018]]).astype(np.float32)
    pts2 = np.array([[926, 735], [2810, 384], [2854, 2231], [904, 2099]]).astype(np.float32)

    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    return matrix

def gamma(img3_path, arc_path, H_alpha, H_beta):
    img3 = cv2.imread(img3_path) # Reading image
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    arch = cv2.imread(arc_path) # Reading image
    arch = cv2.cvtColor(arch,cv2.COLOR_BGR2RGB)

    row, col, _ = img3.shape

    # FOR PERSPECTIVE TRANSFORM
    pts3 = np.array([[926, 735], [2810, 384], [2854, 2231], [904, 2099]]).astype(np.float32)
    ptsarch = np.array([ [0, 0], [1200, 0], [1200, 900], [0, 900]]).astype(np.float32)
    
    matrix = cv2.getPerspectiveTransform(ptsarch, pts3)
    result = cv2.warpPerspective(arch, matrix, (col, row))

    mask = np.zeros(img3.shape, dtype= np.uint8)

    mask_frame = cv2.fillConvexPoly(mask, np.array([pts3], dtype=np.int32), (255,255,255))
    mask_frame = cv2.bitwise_not(mask_frame)

    hollow_frame = cv2.bitwise_and(img3, mask_frame)
    final_frame = cv2.bitwise_or(hollow_frame, result)

    matrix = np.dot(H_alpha, H_beta)
    img1_obtained = cv2.warpPerspective(final_frame, matrix, (col, row))

    plt.imshow(img1_obtained)
    plt.title("Emulation of instructor's image")
    plt.savefig('../results/task_c.png')
    plt.show()

H12 = alpha('../data/1.jpg', '../data/2.jpg')
H23 = beta('../data/2.jpg', '../data/3.jpg')
gamma('../data/3.jpg',"../data/arch.jpg", H12, H23)
# plt.imshow(H12)
# plt.show()
# plt.savefig('../results/task_c.png')

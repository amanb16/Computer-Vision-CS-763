import cv2 
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_pts1():
    return np.array([
        [1512. , 169.], # P
        [1937.,  347.],
        [2341.,  494.],
        [2670.,  618.],
        [2954.,  726.], # Q
        [2971., 1014.],
        [2986., 1300.],
        [3008., 1725.],
        [2998., 2050.], # S
        [2766., 2082.],
        [2393., 2129.],
        [1894., 2192.],
        [1493., 2221.], # R
        [1495., 1804.],
        [1500., 1216.],
        [1516.,  628.],
        
    ], dtype=np.float32)

def get_pts2():
    return np.array([
                    [1321,323],
                    [1692,391],[2218,493],[2725,565],

                    [3005,638],
                    [3005,769],[3012,1193],[3024,1692],

                    [3024,1885],
                    [2785,1908],[2263,1934],[1582,1991],

                    [1306,2002],
                    [1295,1779],[1306,1181],[1325,667]], 
        dtype=np.float32)

def get_pts3():
    return np.array([
        [920,735], # p
        [1242, 667],
        [1734, 595],
        [2354, 459],
        [2793, 383], # q
        [2804,811],
        [2819, 1293],
        [2838, 1855],
        [2846, 2222], # s
        [2316, 2184],
        [1745, 2146],
        [1136, 2104],
        [901, 2078], # r
        [909, 1787],
        [920, 1321],
        [924, 1000]
    ], dtype = np.float32)

def arch_pts(img):
    # print(img.shape)
    h, w, _ = img.shape
    ptsarch=np.array([[0,0], [w/4,0], [w/2,0], [3*w/4, 0],

                      [w,0], [w, h/4], [w ,h/2], [w, 3*h/4],

                      [w, h], [3*w/4, h], [w/2, h], [w/4, h],

                      [0,w], [0, 3*w/4], [0, w/2], [0, w/4]], dtype = np.float32)
    return ptsarch

def alpha(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    row, col, ch= img1.shape

    # P, Q, S, R
    pts1 = get_pts1().astype(np.float32)
    pts2 = get_pts2().astype(np.float32)

    matrix, _ = cv2.findHomography(pts2, pts1)
    # result = cv2.warpPerspective(img2, matrix, (col, row))

    # plt.imshow(result)
    # plt.show()
    return matrix, "a"

def beta(img2_path, img3_path):
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)

    row, col, ch= img2.shape

    pts1 = get_pts2()
    pts2 = get_pts3()

    matrix, mask = cv2.findHomography(pts2, pts1)
    # result = cv2.warpPerspective(img3, matrix, (col, row))


    # plt.imshow(result)
    # # plt.imshow(img3)
    # plt.show()
    return matrix, mask

def gamma(img3_path, arc_path, H_alpha, H_beta):
    img3 = cv2.imread(img3_path) # Reading image
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    arch = cv2.imread(arc_path) # Reading image
    arch = cv2.cvtColor(arch,cv2.COLOR_BGR2RGB)

    row, col, _ = img3.shape

    pts3 = get_pts3()
    ptsarch = arch_pts(arch)

    matrix, _ = cv2.findHomography(ptsarch, pts3)
    result = cv2.warpPerspective(arch, matrix, (col, row))

    mask = np.zeros(img3.shape, dtype= np.uint8)

    mask_frame = cv2.fillConvexPoly(mask, np.array([pts3], dtype=np.int32), (255,255,255))
    mask_frame = cv2.bitwise_not(mask_frame)

    hollow_frame = cv2.bitwise_and(img3, mask_frame)
    final_frame = cv2.bitwise_or(hollow_frame, result)

    matrix = np.dot(H_alpha, H_beta)
    img1_obtained = cv2.warpPerspective(final_frame, matrix, (col, row))

    plt.imshow(img1_obtained)
    plt.savefig('results/task_d.png')
    plt.show()

H12, mask12 = alpha('data/1.jpg', 'data/2.jpg')
H23, mask23 = beta('data/2.jpg', 'data/3.jpg')
gamma('data/3.jpg',"data/arch.jpg", H12, H23)


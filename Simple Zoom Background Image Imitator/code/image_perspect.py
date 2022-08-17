
import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('../data/1.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

arch = cv2.imread('../data/arch.jpg')
arch = cv2.cvtColor(arch,cv2.COLOR_BGR2RGB)

img2 = cv2.imread("../data/2.jpg") # Reading image 2
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img3 = cv2.imread("../data/3.jpg") # Reading image 3
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

global pts3
global ptsarch
global pts2
global pts1
pts3 = np.array([[926, 735], [2810, 384], [2854, 2231], [904, 2099]]).astype(np.float32) # POINTS FOR IMAGE 3
ptsarch = np.array([ [0, 0], [1200, 0], [1200, 900], [0, 900]]).astype(np.float32) # POINTS FOR ARCH
pts2 = np.array([[1329, 340], [3015, 625], [3030, 1893], [1314, 2018]]).astype(np.float32) # POINTS FOR IMAGE 2
pts1 = np.array([[1512, 169], [2954, 726], [2998, 2050], [1493, 2221]]).astype(np.float32) # POINTS FOR IMAGE 1

# THIS FUNCTION WILL PERFORM TASK A
def get_result(img3_path, pts_1, pts_2):

    img3 = cv2.imread(img3_path) # Reading image 
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

    arch = cv2.imread('../data/arch.jpg') # Reading image
    arch = cv2.cvtColor(arch,cv2.COLOR_BGR2RGB)

    row, col, _ = img3.shape

    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
    # matrix, _= cv2.findHomography(pts_1, pts_2)
    result = cv2.warpPerspective(arch, matrix, (col, row))

    mask = np.zeros(img3.shape, dtype= np.uint8)

    mask_frame = cv2.fillConvexPoly(mask, np.array([pts_2], dtype=np.int32), (255,255,255))
    mask_frame = cv2.bitwise_not(mask_frame)

    hollow_frame = cv2.bitwise_and(img3, mask_frame)
    return cv2.bitwise_or(hollow_frame, result)



# plt.imshow(get_result('data/1.jpg', ptsarch, pts1)) 
# plt.savefig('results/instructors_img.png')

plt.figure()
plt.subplot(2,2,2)
plt.imshow(get_result('../data/1.jpg', ptsarch, pts1)) 
# plt.savefig('results/instructors_img.png')
plt.axis('off')
plt.title("Generated Image 1")

plt.subplot(2,2,1)
plt.imshow(arch)
plt.axis('off')
plt.title("Original Image")

plt.subplot(2,2,3)
plt.imshow(get_result('../data/2.jpg', ptsarch, pts2))
plt.axis('off')
plt.title("Generated Image 2")

plt.subplot(2,2,4)
plt.imshow(get_result('../data/3.jpg', ptsarch, pts3))
plt.title("Generated Image 3")
plt.axis('off')
plt.savefig('../results/converted_imgs.png')
plt.show()
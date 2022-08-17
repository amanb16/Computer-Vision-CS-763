
import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('data/tv_1.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

arch = cv2.imread('data/arch_new.jpg')
arch = cv2.cvtColor(arch,cv2.COLOR_BGR2RGB)

img2 = cv2.imread("data/tv_2.png") # Reading image 2
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img3 = cv2.imread("data/tv_3.png") # Reading image 3
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

# h, w, _ = arch.shape

global pts3
global ptsarch
global pts2
global pts1
pts3 = np.array([[178, 19], [296, 18], [292, 80], [182, 81]]).astype(np.float32) # POINTS FOR TV 3
# ptsarch = np.array([ [0,0], [w,0], [w,h], [0,h] ], dtype=np.float32) # POINTS FOR ARCH
ptsarch = np.array([ [0,0], [4128,0], [4128,3096], [0,3096] ], dtype=np.float32) # POINTS FOR ARCH
pts2 = np.array([[346, 8], [605, 174], [605, 384], [346, 564]]).astype(np.float32) # POINTS FOR TV 2
pts1 = np.array([[501, 49], [731,49], [731, 226], [500,182]], dtype=np.float32) # POINTS FOR TV 1

# THIS FUNCTION WILL PERFORM TASK A
def get_result(img3_path, pts_1, pts_2):

    img3 = cv2.imread(img3_path) # Reading image 
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

    arch = cv2.imread('data/arch_new.jpg') # Reading image
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



# plt.imshow(get_result('data/tv_3.png', ptsarch, pts3)) 
# plt.show()
# plt.savefig('results/instructors_img.png')

plt.figure()
plt.subplot(2,2,2)
plt.imshow(get_result('data/tv_1.png', ptsarch, pts1)) 
# plt.savefig('results/instructors_img.png')
plt.axis('off')
plt.title("Generated Image 1")

plt.subplot(2,2,1)
plt.imshow(arch)
plt.axis('off')
plt.title("Original Image")

plt.subplot(2,2,3)
plt.imshow(get_result('data/tv_2.png', ptsarch, pts2))
plt.axis('off')
plt.title("Generated Image 2")

plt.subplot(2,2,4)
plt.imshow(get_result('data/tv_3.png', ptsarch, pts3))
plt.title("Generated Image 3")
plt.axis('off')
plt.savefig('results/converted_taskE.png')
plt.show()
import cv2 
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('../results/instructors_img.png') # READING INSTRUCTOR IMAGE TAKEN FROM TASK a
img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../results/task_c.png') # READING IMAGE CREATED IN TASK C BY GAMMA
img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img3 = cv2.imread('../results/task_d.png') # READING IMAGE CREATED IN TASK D USING HOMOGRAPHY
img_3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

# plt.imshow(img_1)
# plt.show()

# MEAN SQUARE ERROR
def MSE(img1, img2):
    squared_diff = (img1 -img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return np.serr

print("RMSE for task C is: ", MSE(img_1, img_2))
print("RMSE for task D is: ", MSE(img_1, img_3))
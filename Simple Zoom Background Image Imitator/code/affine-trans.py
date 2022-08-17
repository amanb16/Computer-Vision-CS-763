import cv2 
import numpy as np
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-mat',type=str)
args = parser.parse_args()

img1 = cv2.imread("../data/distorted.jpg") # Reading image
cv2.namedWindow('Distorted', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Distorted',650 , 650)

img2 = cv2.imread("../data/figc.jpg") # Reading image

# cv2.circle(img1, (0,0),5, (0, 0, 255), -1) #Top left corner
# cv2.circle(img1, (600,60),5, (0, 0, 255), -1) #Top right corner
# cv2.circle(img1, (660,660),5, (0, 0, 255), -1) #Bottom right corner
# cv2.circle(img1, (60,600),5, (0, 0, 255), -1) #Bottom left corner

pts1= np.array([[0,0], [600,60], [660,660]]).astype(np.float32) #Distorted
pts2= np.array([[0,0], [600,0], [600,600]]).astype(np.float32) # Undistorted
pts3= np.array([[654, 218], [765, 572], [29, 574]]).astype(np.float32) # FOR FIGURE C
l= len(pts1)

a= np.vstack([pts1.T, np.ones(l)])
a= a.T
b= np.linalg.inv(a)
c= np.dot(b, pts2)
A= c.T


if (args.mat=='manual'):
# X'=AX, What we are interested in is A^(-1).
    # A= np.array([[1, -0.1, 0], [-0.1, 1, 0]])
    undistorted = cv2.warpAffine(img1, A, (600,600))

elif(args.mat=="api"):
    a= cv2.getAffineTransform(pts1, pts2)
    undistorted = cv2.warpAffine(img1, a, (600,600))

""" BEGIN OF PART C """
# d= cv2.getAffineTransform(pts3, pts2)
# result= cv2.warpAffine(img2, d, (600,600))

# cv2.imshow("", result)
# # cv2.imwrite('../results/a_figc.png', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
""" END OF PART C """

cv2.imshow("Distorted", img1)
cv2.moveWindow('Distorted',43,50)

cv2.imshow("Undistorted", undistorted)
cv2.moveWindow('Undistorted',680,50)

cv2.waitKey(0)
cv2.destroyAllWindows()


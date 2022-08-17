import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def findKeyPointMatches(desc1,desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.knnMatch(desc1, desc2, 2)
    pruned_matches = []
    n1 = 0.75
    for m,n in matches:
        if m.distance < n1 * n.distance :
            pruned_matches.append(m)
    return pruned_matches
    

def isHomographyPossible(matches, kp1, kp2, dim, MIN_MATCH_COUNT=10, intr = False, img1=None, img2=None):
    if len(matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
        return True, H, mask
    return False, None, None

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def filterInliers(img1, img2, H, matches, kp1, kp2, draw=False):
    inliers1 = []
    inliers2 = []
    good_matches = []
    matched1 = [ kp1[m.queryIdx] for m in matches ]
    matched2 = [ kp2[m.trainIdx] for m in matches ]
    inlier_threshold = 3.5 
    for i, m in enumerate(matched1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt

        col = np.dot(H, col)
        col /= col[2, 0]

        dist = (pow(col[0, 0] - matched2[i].pt[0], 2) + pow(col[1, 0] - matched2[i].pt[1], 2))**0.5
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
        
    if draw:
        res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
        plt.figure(figsize=(15, 5))
        plt.imshow(res)
        plt.show()
    return len(inliers1), len(matched1)
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", type = str, help="input image of lower id with mask to set threshold", default="../replicate/capturedImages/maskLow.jpeg")
    parser.add_argument("-j", type= str, help="image on which check is applied to ", default="../replicate/capturedImages/maskMiddle.jpeg")
    parser.add_argument("-auto", type=str, help="starts with synthetically created hole and keeps expanding the diameter", default="false")
    args = parser.parse_args()
    return args

def setRetrievalScore(database_path="../replicate", maskLow_path="../replicate/capturedImages/maskLow.jpeg"):
    '''
    while checking keep the images of unmasked intruders in database too
    '''
    path=database_path # PATH TO IMAGE DATASET
    imageNames = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):    
            imageNames.append(os.path.join(path, file))

    img1= cv2.imread(maskLow_path) # MASKED MAN IMAGE
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    nfeature = 1500
    sift = cv2.xfeatures2d.SIFT_create(nfeature) 
    kp_1, des_1= sift.detectAndCompute(img1, None)

    inlier_matches = {}
    # LETS ITERATE THROUGH ALL THE IMAGES IN DATASET TO FIND SCORE W.R.T TO ALL IMAGES AND STORE IN 'score' LIST.
    for i in range(len(imageNames)):
        img2= cv2.imread(imageNames[i])
        print(imageNames[i])
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        kp_2, des_2= sift.detectAndCompute(img2, None)
        
        brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        no_of_matches = []
        no_of_matches = brute_force.match(des_1, des_2)
        tmp = findKeyPointMatches(des_1,des_2)
        no_of_matches.extend(tmp)

        no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
        
        yes, H, mask = isHomographyPossible(no_of_matches, kp_1, kp_2, img1.shape )
        if yes and H is not None:
            no_of_matches = np.array(no_of_matches)
            no_of_matches = no_of_matches[np.where(mask.ravel()==1)]
            inlier_count, _ = filterInliers(img1, img2, H, no_of_matches, kp_1, kp_2)
            inlier_matches[imageNames[i]] = inlier_count/sum(mask.ravel())

    inlier_matches_list = sorted([(v,k) for k,v in inlier_matches.items()])[::-1]

    print("\nPrinting top 10 retrieval (Best Match Topmost)")

    for i,vim in enumerate(inlier_matches_list[:10]):
        print(i+1, vim) 

    print("\nSetting retrieval threshold to 0.65")
    print("\nIs the intruder present?")
    if inlier_matches_list[0][0] >= 0.65:
        print("Yes, Intruder present in the database")
    else:
        print("Nope, Intruder not found, seems like more matching to do")


def findIntruderMiddle(srcpath = "../replicate/capturedImages/Middle.jpeg", maskedImgpath="", single=False):
    image = cv2.imread(srcpath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    masked = None
    if single:
        masked = cv2.imread(maskedImgpath)
        masked = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)

    nfeature = 1500
    sift = cv2.xfeatures2d.SIFT_create(nfeature) 

    kp_1, des_1= sift.detectAndCompute(image, None)
    k=30
    i = 1
    open("../replicate/output/retScores.txt", "w").close()
    retFile = open("../replicate/output/retScores.txt", "a")
    while True:
        if not single:
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.circle(mask, (int(250), int(350)), k, 255, -1)
            k=k+10
            masked = cv2.bitwise_and(image, image, mask=mask)
            masked[np.where((masked==[0,0,0]).all(axis=2))] = [255,255,255]

        # masked = cv2.resize(masked, (600,700))
        # masked = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
        kp_2, des_2= sift.detectAndCompute(masked, None)

        brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) #Or L1
        no_of_matches = brute_force.match(des_1, des_2)
        no_of_matches.extend(findKeyPointMatches(des_1,des_2))
        no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
        yes, H, mask = isHomographyPossible(no_of_matches, kp_1, kp_2, image.shape )
        s = ""
        ans = 0
        if yes and len(no_of_matches)>20 and H is not None:
            '''
            if you wish to plot image matches set draw parameter below to True
            '''
            inlier_count, total_inlier = filterInliers(image, masked, H, no_of_matches, kp_1, kp_2, draw=False)
            rt_score = inlier_count/total_inlier
            print("Diameter:{}, Score: {}".format(2*k,rt_score))
            s = "maskMiddle{}.png\t{}\n".format(i,rt_score)
            retFile.write(s)
            if rt_score >= 0.65: 
                print("Threshold crossed. Stopping...")
                cv2.imwrite("../replicate/output/maskMiddleBest.png", cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
                break
            cv2.imwrite("../replicate/capturedImages/maskMiddle{}.png".format(i), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
            i+=1
            if k > max(image.shape[:2]): break
            ans = 1
        if single:
            if not ans: print("Diameter not sufficient, please increase")
            break
    retFile.close()

'''
auto parameter will increase diameter automatically starting from a very small hole.
all intermediate images gets saved in captured ../data/replicate/capturedImages directory 
all final is saved in ../data/replicate/output directory 
'''

args = get_args()
print("\nProcessing images in database\n")
setRetrievalScore(maskLow_path=args.i)

print("\nFinding diameter for image\n")
single = False if args.auto.lower()=="true" else True
findIntruderMiddle(maskedImgpath=args.j, single=single)
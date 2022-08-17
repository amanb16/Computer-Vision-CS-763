import argparse
import cv2
import numpy as np
from Utils.VideoUtils import VideoReader, VideoWriter
from Utils.common_scripts import externalHomography, findMatches, getFeatures, getPoints
from Utils.flowUtility import readFlow

def get_args():
    '''
    Description: 
        set command line arguments
    Parameters:
        None
    Returns: 
        argparser object
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, help="question No.")
    parser.add_argument('--subQuestion',type=str, help="subquestion A or B")
    parser.add_argument('--video', type=int, help="video no.")

    
    return parser.parse_args()

def flow_utils(video, apply_mask=False, mask_write=None, given_flow=False):
    
    flow = np.zeros((video.nrFrames-1, 3), np.float32)
    prev = video.getNextFrame()
    curr = video.getNextFrame()

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape

    for n in range(249):
        prev_pts = np.zeros((360*640+1,2))
        curr_pts = np.zeros((360*640+1,2))
        dx, dy = np.zeros((360*640+1,1)), np.zeros((360*640+1,1))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Optical flow is now calculated

        if given_flow:
            flow_dup = video.getFlow(n)
        else:
            flow_dup = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5,3, 15, 3, 5, 1.2, 0)



        for i in range(1,h+1):
            for j in range(1, w+1):
                prev_pts[i*j,0], prev_pts[i*j,1] = i,j
                dx[i*j,0],dy[i*j,0] = flow_dup[i-1,j-1,0], flow_dup[i-1,j-1,1]
                curr_pts[i*j,0], curr_pts[i*j,1] = prev_pts[i*j,0]+dx[i*j,0], prev_pts[i*j,1]+dy[i*j,0]
                
        prev_pts = np.delete(prev_pts, 0, 0)
        curr_pts = np.delete(curr_pts, 0, 0)
        #print(curr_pts[0])
        #print(prev_pts[0])

        if apply_mask:
            min_a, max_a = -1e-7, np.inf
            mask = np.clip(flow_dup, min_a, max_a)
            mask = np.all(mask == min_a, axis=-1).astype(np.uint8)*255


        # filter for better matches
        H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=3.0)

        # store the changes
        angle = np.arctan2(H[1,0], H[0,0])
        flow[n] = [H[0,2],H[1,2],angle]
        print("On frame No.:",n)
        # k+=1
        if apply_mask:
            mask_write.writeFrame(np.stack((cv2.hconcat([curr,mask]),)*3, axis=-1))
        #move to next frames here
        prev = curr.copy()
        curr = video.getNextFrame()
        #print(flow.shape)

    return flow

def flow_1(n):
	flow = []
	i = -1
	while True:
		if i<9:
			i+=1
			path = f"data/Q2/1_flows/000{str(i)}.flo"
			f = readFlow(path)
			flow.append(f)
		elif 8<i<99:
			i+=1
			path = f"data/Q2/1_flows/00{str(i)}.flo"
			f = readFlow(path)
			flow.append(f)
		elif 98<i<248:
			i+=1
			path = f"data/Q2/1_flows/0{str(i)}.flo"
			f = readFlow(path)
			flow.append(f)
		else :
			break
	return flow[n]

        
	
def getFLowB(video, apply_mask=False, mask_write=None, given_flow=False):
    '''
    Description:
        finds matches between two consecutive frames
        Using homography find change in x,y and angle 
    Parameters:
        video object
    Returns:
        flow (np.array): changes between 2 consecutive images hence size no.frame-1
    
    '''
    flow = np.zeros((video.nrFrames-1, 3), np.float32)
    prev = video.getNextFrame()
    curr = video.getNextFrame()
    i = 0
    height, width = prev.shape[0], prev.shape[1]
    coordinate_mesh = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))#np.concatenate((x[...,np.newaxis],y[...,np.newaxis]), axis=-1)
    while curr is not None:

        curr_copy = curr.copy()
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        if given_flow:
            opticalFLow = video.getFlow(i)
        else:
            opticalFLow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5,3, 15, 3, 5, 1.2, 0)
        
        if apply_mask:
            min_a, max_a = -1e-7, np.inf
            mask = np.clip(opticalFLow, min_a, max_a)
            mask = np.all(mask == min_a, axis=-1).astype(np.uint8)*255
            prev_pts = coordinate_mesh[np.where(mask==255)]
            curr_pts = prev_pts + opticalFLow[np.where(mask==255)]
            prev_pts = np.float32(prev_pts)
            curr_pts = np.float32(curr_pts)
        else:
            prev_pts = coordinate_mesh.copy()
            curr_pts = prev_pts + opticalFLow
            prev_pts = np.float32(prev_pts)
            curr_pts = np.float32(curr_pts)
        
        # height, width = opticalFLow.shape[0], opticalFLow.shape[1]
        # R2 = coordinate_mesh.copy()
        # pixel_map = R2 + opticalFLow
        # new_frame = cv2.remap(prev, np.float32(pixel_map), None, interpolation =cv2.INTER_LINEAR)

        # windowName = "stable"
        bin_mask = cv2.hconcat([curr,mask])

        if apply_mask:
            mask_write.writeFrame(np.stack((bin_mask,)*3, axis=-1))
        # k = cv2.waitKey(10)
        # if k == ord('q'): cv2.destroyAllWindows();break

        # mag, ang = cv2.cartToPolar(opticalFLow[..., 0], opticalFLow[..., 1])
        try:
            H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=0.0001)
            angle = np.arctan2(H[1,0], H[0,0])
            flow[i] = [H[0,2],H[1,2],0]
        except:
            flow[i] = flow[i-1]
        print("On Frame No:",i)
        # print(opticalFLow[np.where(mask==255)].shape)
        # avg_u = np.mean(opticalFLow[np.where(mask==255)][ :, 0])
        # avg_v = np.mean((opticalFLow[np.where(mask==255)])[ :, 1])
        # angle = np.mean(ang[np.where(mask==255)])*180/np.pi
        # # print(avg_u)
        # flow[i] = [avg_u,avg_v,angle]
        # i+=1
        # H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=3.0)
        # store the changes
        
        i+=1
        # move to next frames
        prev = curr_copy
        curr = video.getNextFrame()
    return flow

def getFLowA(video):
    '''
    Description:
        finds matches between two consecutive frames
        Using homography find change in x,y and angle 
    Parameters:
        video object
    Returns:
        flow (np.array): changes between 2 consecutive images hence size no.frame-1
    
    '''
    flow = np.zeros((video.nrFrames-1, 3), np.float32)
    prev = video.getNextFrame()
    curr = video.getNextFrame()
    i = 0
    height, width = prev.shape[0], prev.shape[1]
    while curr is not None:
        # get point correspondences
        curr_copy = curr.copy()
        kp_p, des_p = getFeatures(prev)
        kp_c, des_c = getFeatures(curr)
        matches = findMatches(des_p, des_c)
        prev_pts, curr_pts, _, _ = getPoints(kp_p, kp_c, matches)


        # filter for better matches
        try:
            H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=3.0)

            # store the changes
            angle = np.arctan2(H[1,0], H[0,0])
            flow[i] = [H[0,2],H[1,2],angle]
        except:
            flow[i] = [0,0,0]
        i+=1
        print("On Frame No:",i)
        
        
        # move to next frames
        prev = curr_copy
        curr = video.getNextFrame()
    return flow

def gaussian_filter1d(size,sigma):
    '''
    Description:
        Generates 1D gaussian filter of given size and standard deviation
    Parameters:
        size (int): size of the filter
        sigma (float): standard deviation of gaussian (higher value give smoother gaussians)
    Returns:
        gaussian_filter (np.array): gaussian filter
    '''
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return gaussian_filter

def convolveGaussianFilter(base_array, size, sigma=10):
    '''
    Description:
        pad and convolve gaussian filter to smooth given array
    Parameters:
        base_array (np.array): array over which smoothing is to be done
        size (int): size of the gaussian filter
        sigma (float): standard deviation of gaussian (higher value give smoother gaussians)
    Returns:
        base_array_filtered (np.array): smoothed base array
    '''
    gaussian_filter = gaussian_filter1d(size, sigma)
    base_array = np.pad(base_array, (size, size), 'edge')
    base_array_filtered = np.convolve(base_array, gaussian_filter, mode='same')
    base_array_filtered = base_array_filtered[size:-size]
    return base_array_filtered

def smooth(trajectory):
    '''
    Description:
        apply smoothing using gaussian filter over all 3 axis x,y,angle separately
    Parameters:
        trajectory (np.array): motion/flow array to be smoothed
    Returns:
        smoothed_trajectory (np.array): smoothed given motion/flow array
    '''
    smoothed_trajectory = np.copy(trajectory)
    r = smoothed_trajectory.shape[0]
    print(r) #No. of frames
    for i in range(3):
        smoothed_trajectory[:,i] = convolveGaussianFilter(trajectory[:,i], int(np.ceil(r*0.25)), sigma=15)
    return smoothed_trajectory  

def smoothFlow(flow):
    '''
    Description:
        apply smoothing over the flow using 1st frame as reference
    Parameters:
        flow (np.array): motion/flow array to be smoothed
    Returns:
        transformed_flow (np.array): smoothed flow 
    '''
    cumulative_flow = np.cumsum(flow, axis=0)
    smoothed_flow = smooth(cumulative_flow)
    difference = smoothed_flow - cumulative_flow
    transformed_flow = flow + difference
    return transformed_flow

def bgFLow(video):
    '''
    Description:
        distinguish background
    Parameters:
        video object
    Returns:
        flow (np.array): changes between 2 consecutive images hence size no.frame-1
    
    '''
    flow = np.zeros((video.nrFrames-1, 3), np.float32)
    prev = video.getNextFrame()
    curr = video.getNextFrame()
    i = 0
    while curr is not None:
        kp_p, des_p = getFeatures(prev)
        kp_c, des_c = getFeatures(curr)
        matches = findMatches(des_p, des_c)
        prev_pts, curr_pts, _, _ = getPoints(kp_p, kp_c, matches)

        # filter for better matches
        H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=3.0)

        # move to next frames
        res1 = cv2.warpPerspective(prev, H, (video.width,video.height))
        res2 = cv2.warpPerspective(curr, H, (video.width,video.height))
        diff1 = cv2.absdiff(res1, curr)
        diff2 = cv2.absdiff(res2, curr)
        mask1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
        mask2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
        th = 30
        imask1 =  mask1>th
        kmask1 = mask1<th
        imask2 =  mask2>th
        kmask2 = mask2<th
        canvas1 = np.zeros_like(res1, np.uint8)
        canvas1[imask1] = 0
        canvas1[kmask1] = 255
        bg1 = res1 - canvas1 #or curr - canvas #Take it has prev
        canvas2 = np.zeros_like(res2, np.uint8)
        canvas2[imask2] = 0
        canvas2[kmask2] = 255
        bg2 = res2 - canvas2
        prev = bg1
        curr = bg2
        # get point correspondences
        kp_p, des_p = getFeatures(prev)
        kp_c, des_c = getFeatures(curr)
        matches = findMatches(des_p, des_c)
        prev_pts, curr_pts, _, _ = getPoints(kp_p, kp_c, matches)
        # filter for better matches
        H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=3.0)
        # store the changes
        angle = np.arctan2(H[1,0], H[0,0])
        flow[i] = [H[0,2],H[1,2],angle]
        i+=1
        # move to next frames
        prev = curr.copy()
        curr = video.getNextFrame()
        return flow

def stabilize(question, subQuestion, video, out_video, mask_res):
    '''
    Description:
        calculate flow, smooth it, and use it to stabilize a given video, and write the video
    Parameters:
        video (VideoReader Object): video object to read and perform stabilization on
        out_video (VideoWriter Object): write the stabilized video
    '''
    #  get flow of a video
    if (question == '1' or question=='3' )and subQuestion == 'A':
        flow = getFLowA(video)
    elif (question == '1'  or question=='3') and subQuestion == 'B':
        flow = getFLowB(video)
    elif question == '2' and subQuestion == 'A':
        featurepoints(video, mask_res)
        video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flow = bgFLow(video)
    elif question == '2' and subQuestion == 'B':
        flow = flow_utils(video, apply_mask=True, mask_write=mask_res, given_flow=False)
    elif question == '2' and subQuestion =='C':
        video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flow = flow_utils(video, apply_mask=True, mask_write=mask_res, given_flow=True)
    elif question == '2' and subQuestion == 'D':
        flow = getFLowA(video)
    elif question == '2' and subQuestion == 'E':
        flow = getFLowB(video)


    transforms_smooth = smoothFlow(flow)


    # reset reading head to apply stabilize from 1st frame
    video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    curr = video.getNextFrame()
    i = 0
    while curr is not None and (i<video.nrFrames-1):
        x = transforms_smooth[i,0]
        y = transforms_smooth[i,1]
        angle = transforms_smooth[i,2]
        i+=1
        # affine matrix will do
        # if want to do warpPerspective 
        # add [0,0,1] at the very end same effect as
        # only translation and rotation is done
        T = np.array([
            [np.cos(angle),-np.sin(angle),x],
            [np.sin(angle),np.cos(angle),y],
        ])

        frame_stabilized = cv2.warpAffine(curr, T, (video.width,video.height))
        # frame_stabilized = fixBorder(frame_stabilized)
        frame_out = cv2.hconcat([curr, frame_stabilized])

        # windowName = "stable"
        # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        # cv2.imshow(windowName, frame_out)
        out_video.writeFrame(frame_out)
        k = cv2.waitKey(10)
        if k == ord('q'): cv2.destroyAllWindows();break
        curr = video.getNextFrame()
        
def featurepoints(video, out_video):
    '''
    Description:
        shows green and red feature points in video
    Parameters:
        video object
    Returns:
        flow (np.array): saves the video
    
    '''
    
    prev = video.getNextFrame()
    curr = video.getNextFrame()
    i = 0
    while curr is not None:
        kp_p, des_p = getFeatures(prev)
        kp_c, des_c = getFeatures(curr)
        matches = findMatches(des_p, des_c)
        prev_pts, curr_pts, _, _ = getPoints(kp_p, kp_c, matches)

        # filter for better matches
        H, _ = externalHomography(prev_pts, curr_pts, useCVRansac=True, threshold=3.0)

        # move to next frames
        res = cv2.warpPerspective(prev, H, (video.width,video.height))

        diff = cv2.absdiff(res, curr)

        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        th = 30
        imask =  mask>th
        kmask = mask<th
        canvas = np.zeros_like(curr, np.uint8)
        canvas[imask] = 0
        canvas[kmask] = 255
        bg = res - canvas #or curr - canvas
        
        kp1, des1 = getFeatures(canvas)
        kp2, des2 = getFeatures(bg)
        out1 = cv2.drawKeypoints(curr, kp2, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        out2 = cv2.drawKeypoints(out1, kp1, 0, (255, 0, 0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        frame_out = cv2.hconcat([curr, out2])
        
        # windowName = "feature_points"
        # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        # cv2.imshow(windowName, frame_out)
        out_video.writeFrame(frame_out)
        k = cv2.waitKey(10)
        if k == ord('q'): cv2.destroyAllWindows();break
        prev = curr.copy()
        curr = video.getNextFrame()

if __name__ == "__main__":
    '''
    Please create the output directories
    '''
    print("Begins... Takes a little time")
    args = get_args()
    video = VideoReader("../data", args.question, args.video, loadAllFrames=False)
    out_video = VideoWriter("../Output/{}_{}/{}.avi".format(args.question, args.subQuestion, args.video))
    mask_res = VideoWriter("../Output/{}_{}/binary_mask_{}.avi".format(args.question, args.subQuestion, args.video))
    stabilize(args.question, args.subQuestion, video, out_video, mask_res)
    print("Done...")


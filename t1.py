#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    sift = cv2.xfeatures2d.SIFT_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Extract a set of key points for each image.
    #Extract features from each key point.
    keypoint1, descriptor1 = sift.detectAndCompute(gray1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(gray2, None)

    #Match features and use matches to determine if there is overlap between given pairs of images.
    #1-Brute Force Matcher
    match = []
    idx=[]
    for i in range(len(descriptor1)):
        dist = []
        for j in range(len(descriptor2)):
            l2norm = np.sqrt(np.sum((descriptor1[i]-descriptor2[j])**2))
            dist.append(l2norm)
        sorted_dist = sorted(dist)
        match.append([sorted_dist[0],sorted_dist[1]])
        idx.append([i,dist.index(min(dist))])


    #2-find good matches
    good_matches = []
    for i in range(len(match)):
        if match[i][0] < match[i][1]*0.2:
            good_matches.append([idx[i][0],idx[i][1]])

    matches = np.array(good_matches)
    #Compute the homography between the overlapping pairs as needed. Use RANSAC to optimize your result.
    point1 = []
    point2 = []
    if (len(matches[:,0]) >= 4):
        for i in good_matches:
            point1.append(keypoint1[i[0]].pt)
            point2.append(keypoint2[i[1]].pt)
        
        point1 = np.array(point1)
        point2 = np.array(point2)

        matrix, _ = cv2.findHomography(point1, point2, cv2.RANSAC,5.0)
    else:
        raise AssertionError("can't find enough keypoints.")

    #Transform the images and stitch the two images into one mosaic, eliminating the foreground as described above, but do NOT crop your image.
    #1-eliminating the foreground
    mask = np.zeros(img2.shape[:2],np.uint8)

    bgModel = np.zeros((1,65),np.float64)
    fgModel = np.zeros((1,65),np.float64)   

    rect=(390,100,210,400)
    cv2.grabCut(img2,mask,rect,bgModel,fgModel,1,cv2.GC_INIT_WITH_RECT)

    newmask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    cropimg = img2*newmask[:,:,np.newaxis]

    crop = np.copy(img2-cropimg)

    #2-stich the two imgs into one mosaic
    width = img1.shape[1] + img2.shape[1] 
    height = img1.shape[0] + img2.shape[0]
    
    out = cv2.warpPerspective(img1, matrix,(width,height))
    out[0:img2.shape[0], 0:img2.shape[1]] = crop

    plt.figure(figsize=(20,10))
    plt.imshow(out)
    plt.show()
    #Save the resulting mosaic according to the instructions specified in the script. 
    cv2.imwrite(savepath,out)

    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)


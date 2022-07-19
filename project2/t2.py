# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

import json

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."

    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    img1=imgs[0]
    img2=imgs[1]
    img3=imgs[2]
    img4=imgs[3]

    def extract(img):
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoint,descriptor = sift.detectAndCompute(gray,None)
        return keypoint,descriptor
    
    #Match features and use matches to determine if there is overlap between given pairs of images.
    def matching_matrix(descriptor1,descriptor2,keypoint1,keypoint2):
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

        #Compute the homography between the overlapping pairs as needed. Use RANSAC to optimize your result.

        point1 = []
        point2 = []
        for i in good_matches:
            point1.append(keypoint1[i[0]].pt)
            point2.append(keypoint2[i[1]].pt)
        
        point1 = np.array(point1)
        point2 = np.array(point2)

        matrix, _ = cv2.findHomography(point1, point2, cv2.RANSAC)

        return good_matches,matrix

    #Compute the overlap arrays 
    overlap_arr = np.zeros((N,N))
    for i in range(overlap_arr.shape[0]):
        for j in range(overlap_arr.shape[1]):
            if i==j:
                print("i=j just overlap it")
                overlap_arr[i][j] = 1
            else:
                print("Start extracting and matching features (%d, %d)"%(i,j))
                keypoint_1, descriptor_1 = extract(imgs[i])
                keypoint_2, descriptor_2 = extract(imgs[j])
                try:
                    g, m = matching_matrix(descriptor_1,descriptor_2,keypoint_1,keypoint_2)
                    if len(g) > 20:
                        overlap_arr[i][j] = 1
                except:
                    print('no matching with (%d, %d)'%(i,j))
                if i == 0:
                    dst = cv2.warpPerspective(imgs[i],m,((imgs[i].shape[1] + imgs[j].shape[1]), (imgs[i].shape[0]+imgs[j].shape[0]))) #wraped image
                    dst[0:imgs[j].shape[0], 0:imgs[j].shape[1]] = imgs[j] #stitched image
                else:
                    dst = cv2.warpPerspective(dst,m,((dst.shape[1] + imgs[j].shape[1]), (dst.shape[0]+imgs[j].shape[0]))) #wraped image
                    dst[0:imgs[j].shape[0], 0:imgs[j].shape[1]] = imgs[j] #stitched image
          

    print(overlap_arr)
                
    #Transform the images and stitch the two images into one mosaic, eliminating the foreground as described above, but do NOT crop your image.
    """
    keypoint1, descriptor1 = extract(img1)
    keypoint2, descriptor2 = extract(img2)
    keypoint3, descriptor3 = extract(img3)
    keypoint4, descriptor4 = extract(img4)


    g1,m1 = matching_matrix(descriptor1,descriptor2,keypoint1,keypoint2)
    g2,m2 = matching_matrix(descriptor2,descriptor3,keypoint2,keypoint3)
    g3,m3 = matching_matrix(descriptor3,descriptor4,keypoint3,keypoint4)

    width = img1.shape[1] + img2.shape[1]  
    height = img1.shape[0] + img2.shape[0]  
    
    dst = cv2.warpPerspective(img1, m1,(width,height),dst=img1.copy(),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_TRANSPARENT)
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    """
    """
    width = width + img3.shape[1]
    height = height + + img3.shape[0]
    
    out2 = cv2.warpPerspective(out1, m2,(width,height),dst=img2.copy(),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_TRANSPARENT)
    out2[0:img3.shape[0], 0:img3.shape[1]] = img3
    
    width = width + img4.shape[1]
    height = height + + img4.shape[0]

    out3 = cv2.warpPerspective(img4, m3,(width,height),dst=img3.copy(),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_TRANSPARENT)
    out3[0:out2.shape[0], 0:out2.shape[1]] = out2
     """
    

    plt.figure(figsize=(20,10))
    plt.imshow(dst)
    plt.show()
    cv2.imwrite(savepath,dst)
    
    return overlap_arr

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)

'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
#from helper import show_image

import cv2
import numpy as np
import os
import sys

#import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    for idx in range(1, 104):
        f_name = 'img_'+str(idx)+'.jpg'
        img = cv2.imread(input_path+'/'+f_name)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        locs = cascade.detectMultiScale(gray,1.1,5)
        for (x,y,w,h) in locs:
            bbox = [int(x), int(y), int(w), int(h)]
            result_list.append({'iname':f_name, 'bbox':bbox})
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    
    face_encoded_list = []
    
    for idx in range(1, 37):
        f_name = str(idx)+'.jpg'
        img = face_recognition.load_image_file(input_path+f_name)
        boxes = face_recognition.face_locations(img) #top right bottom left 
        face_encoded = face_recognition.face_encodings(img, boxes)
        face_encoded_list.append(face_encoded)

    center_idx = np.random.choice([i for i  in range(len(face_encoded_list))], K, replace=False)
    result_list = [0 for i in range(len(face_encoded_list))]
    change = len(result_list)

    while True:
        for idx, face in enumerate(face_encoded_list):
            min = float("inf")
            for c_idx, c in enumerate(center_idx):
                # l2 distance
                center = face_encoded_list[c]
                dist = np.sqrt(np.sum(np.square(np.array(face)-np.array(center))))
                if dist < min:
                    min = dist
            if c != result_list[idx]:
                change -= 1
                result_list[idx] = c_idx
            
            if change == len(result_list):
                break

    return result_list

'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""

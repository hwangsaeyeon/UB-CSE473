"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    testImg, chr = enrollment(test_img, characters)
    
    #save the features 
    import pandas as pd
    for k in range(len(chr)):
      data=[]
      for i in range(len(chr[k])):
        data.append(chr[k][i])
      df = pd.DataFrame([data],columns=['character','features'])
      df.to_csv('./features/'+chr[k][0]+'_features.csv',index=False)
    
    labeled = detection(testImg)
    #recognition(chr)

    #raise NotImplementedError

def enrollment(test_img,characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """


    # TODO: Step 1 : Your Enrollment code should go here.

    #generate features in test_img
    test_array = []
    for i in range(len(test_img)):
      for j in test_img[i]:
        test_array.append([j,j,j])
    test_array = np.array(test_array)
    test = np.reshape(test_array, (len(test_img),len(test_img[i]),3))
    test = test.astype('uint8')
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #feature colors are black, background colors are white
    cv2.imwrite('test_binary.png',dst)
    test_img = dst


    #generate features in characters
    for k in range(len(characters)):
      chr = characters[k][1]

      img_array = []
      for i in range(len(chr)):
        for j in chr[i]:
          img_array.append([j,j,j])
      img_array = np.array(img_array)
      chr = np.reshape(img_array,(len(chr),len(chr[i]),3))
      chr = chr.astype('uint8')

      gray = cv2.cvtColor(chr, cv2.COLOR_BGR2GRAY)
      ret, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
      edges = cv2.Canny(dst,100,200)
      cv2.imwrite(characters[k][0]+'_canny.png',edges)
      characters[k][1] = edges
  
    return test_img, characters

    #raise NotImplementedError

def detection(test):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    
    #restrict inputs to binary (black and white) images. 
    #Background pixels will be labelled '0' but this code, '255' 
    #done at enrollment part 

    x,y = test.shape
    labeled_img = np.array(np.ones([x,y]))
    label_cnt = 200

    #first pass
    for i in range(len(test)):
      for j in range(len(test[0])):

        left = test[i][j-1]
        up = test[i-1][j]

        if test[i][j] > 0 : #background img  #feature colors are black(0), background colors are white(255) in test img 
          pass 

        else: #foreground img  
          if left > 0  and up > 0 : #no neighbors = edge of top and left 
            labeled_img[i][j] = label_cnt
            label_cnt += 1
          elif left == 0 and up > 0: #LEFT neighbor
            labeled_img[i][j] = labeled_img[i][j-1] 
          elif left > 0 and up == 0: #TOP neighbor
            labeled_img[i][j] = labeled_img[i-1][j]
          elif left == 0 and up == 0: #multiple neighbors
            labeled_img[i][j] = min(labeled_img[i-1][j],labeled_img[i][j-1])

    #second pass 
    for i in range(len(test)):
      for j in range(len(test[0])):

        if labeled_img[i][j] > 0 : 
          if (j+1 != len(test[0])) and (test[i][j] == test[i][j+1]) : 
            #if j is not out of bound 
            #if test is foreground and join on the right side 
            if labeled_img[i][j] != labeled_img[i][j+1]:
              labeled_img[i][j] = min(labeled_img[i][j],labeled_img[i][j+1])

    cv2.imwrite('labeled_img.png',labeled_img)

    return labeled_img

    #raise NotImplementedError


def recognition(chr):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()

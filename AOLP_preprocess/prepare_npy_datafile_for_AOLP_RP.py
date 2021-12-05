import os
from os import path
import numpy as np
from numpy import save
import cv2


class_dict = {  '0':0,
                '1':1,
                '2':2,
                '3':3,
                '4':4,
                '5':5,
                '6':6,
                '7':7,
                '8':8,
                '9':9,
                'A':10,
                'B':11,
                'C':12,
                'D':13,
                'E':14,
                'F':15,
                'G':16,
                'H':17,
                'J':18,
                'K':19,
                'L':20,
                'M':21,
                'N':22,
                'P':23,
                'Q':24,
                'R':25,
                'S':26,
                'T':27,
                'U':28,
                'V':29,
                'W':30,
                'X':31,
                'Y':32,
                'Z':33
                } 
                


# preprocess image data

def load_image(file):
    try: 
        img = cv2.imread(file)
        img2 = cv2.resize(img, (160,64))
        img_arr = np.array(img2)
        
    except:
        print("read image file error")
        exit(0)
    return img_arr


def load_label(file):
    plate = []
    with open(file) as f_1 :
        lines = f_1.readlines()
        lines = [line.rstrip() for line in lines]
        
        label_in_num = []
        char_num = 0
        for char in lines:
            char_num += 1                
            if char_num > 6:
                print(file+" has more than 6 chars or empty line in the end")
                exit(0)
            
            if not char.isalnum(): 
                print(file+" has a special char or length of label chars < 6")
                exit(0)
            if char == 'O': 
                print(file+" has char 'O'")
                exit(0)
            if char == 'I': 
                print(file+" has char 'I'")
                exit(0)    
                
            num_1 = class_dict[char]
            label_in_num.append(num_1)

        if char_num < 6:
                print(file1+" has less than 6 chars")
                exit(0)            

        return label_in_num



img_dir = "data/AOLP_RP_plates/"
label_dir = "data/AOLP_RP_labels/"

file_list = os.listdir(img_dir)

images = []
labels = []

for file in file_list:
    file_name_p1,file_name_p2 = os.path.splitext(file)
    
    img_file = img_dir+file
    images.append(load_image(img_file))
    
    label_file = label_dir+file_name_p1+".txt"
    labels.append(load_label(label_file))
    

images = np.array(images)
labels = np.array(labels)

save('data/AOLP_RP_plate_images.npy', images)
save('data/AOLP_RP_plate_labels_in_number.npy', labels)  
  

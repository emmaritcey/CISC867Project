import cv2
import os

def plate_extractor(imgs_dir, boxes_dir, plates_dir):  
    contents_img = os.listdir(imgs_dir)
    contents_box = os.listdir(boxes_dir)

    for img_name in contents_img:
        name1,name2 = os.path.splitext(img_name)
        img_raw = cv2.imread(imgs_dir+img_name, cv2.IMREAD_COLOR)
        print(img_name)
        file1 = name1+"_2.txt"

        with open(boxes_dir+file1) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            box1 = []
            for line in lines:
                n1 = float(line)
                n1 = int(n1)
                box1.append(n1)
              
        x1 = box1[0]
        y1 = box1[1]
        x2 = box1[2]
        y2 = box1[3]
        img = img_raw[y1:y2,x1:x2]
        img = cv2.resize(img, (160, 50))
        cv2.imwrite(plates_dir+img_name,img)
                
    return

imgs_dir = "./AOLP_LE2_Image/"
boxes_dir = "./AOLP_LE2_groundtruth_localization/"
plates_dir = "./AOLP_LE2_plates/"
if not os.path.isdir(plates_dir):
    os.mkdir(plates_dir) 
plate_extractor(imgs_dir, boxes_dir, plates_dir)


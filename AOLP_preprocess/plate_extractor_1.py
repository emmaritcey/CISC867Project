import cv2
import os

def plate_extractor(imgs_dir, boxes_dir, plates_dir):  
    contents_img = os.listdir(imgs_dir)
    contents_box = os.listdir(boxes_dir)

    for img_name in contents_img:
        name1,name2 = os.path.splitext(img_name)
        img_raw = cv2.imread(imgs_dir+img_name, cv2.IMREAD_COLOR)
        if(img_name == "Thumbs.db"): continue
        print(img_name)
        file1 = name1+".txt"

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
        if(x1>x2):
            x1 = box1[2]
            x2 = box1[0]    
        if(y1>y2):
            y1 = box1[3]
            y2 = box1[1]           
        img = img_raw[y1:y2,x1:x2]
        img = cv2.resize(img, (160, 50))
        cv2.imwrite(plates_dir+img_name,img)

                
    return

  
imgs_dir = "./AOLP_AC_Image/"
boxes_dir = "./AOLP_AC_groundtruth_localization/"
plates_dir = "./AOLP_AC_plates/"
if not os.path.isdir(plates_dir):
    os.mkdir(plates_dir) 
plate_extractor(imgs_dir, boxes_dir, plates_dir)


imgs_dir = "./AOLP_LE_Image/"
boxes_dir = "./AOLP_LE_groundtruth_localization/"
plates_dir = "./AOLP_LE_plates/"
if not os.path.isdir(plates_dir):
    os.mkdir(plates_dir) 
plate_extractor(imgs_dir, boxes_dir, plates_dir)


imgs_dir = "./AOLP_RP_Image/"
boxes_dir = "./AOLP_RP_groundtruth_localization/"
plates_dir = "./AOLP_RP_plates/"
if not os.path.isdir(plates_dir):
    os.mkdir(plates_dir) 
plate_extractor(imgs_dir, boxes_dir, plates_dir)

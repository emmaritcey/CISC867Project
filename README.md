

>📋  A template README.md for code accompanying a Machine Learning paper

# CISC867Project

This repository is the reproduced implementation of [Efficient License Plate Recognition via Holistic Position Attention](https://ojs.aaai.org/index.php/AAAI/article/view/16457). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data Preparation and Preprocessing
You need to download the [AOLP](http://aolpr.ntust.edu.tw/lab/) datasets.<br>

The plates char need to be converted to pure number format<br>

All the image files and label files can be saved in npy files<br>
<pre>
├── data
│     ├── AOLP_noSkew
│     │     ├── label_in_number
│     │     └── image
│     │   
│     ├── AOLP_AC_plate_images.npy  
│     ├── AOLP_AC_plate_labels_in_number.npy
│     ├── AOLP_LE_plate_images.npy  
│     ├── AOLP_LE_plate_labels_in_number.npy  
│     ├── AOLP_RP_plate_images.npy  
│     ├── AOLP_RP_plate_labels_in_number.npy  
│     ├── AOLP_noSkew_plate_images.npy  
│     └── AOLP_noSkew_plate_labels_in_number.npy

</pre>

Use plate_extractor1.py or plate_extractor2.py to extract just the license plate from the AOLP images

## Training
The model architecture can be viewed in LPRmodel.py <br>
    - line 20: choose base model to use (resnet101, 50, 34, or 18)

To train, run: <br>
python lpr_train <br>
    - lines 12-17: alter datasets to train on prior to running if needed 


## Evaluation
To test and evaluate the trained model, run: <br>
python lpr_test <br>
    - lines 11,12: load appropriate dataset to test on (test on third AOLP dataset not used for training)


## Pre-trained Models
The model uses a ResNet model pretrained on ImageNet dataset. Can choose to use either a resnet101, resnet50, resnet34, resnet18 pretrained on ImageNet. Specified in LPRmodel.py on line 20


## Results

|     Model name     |    AC    |    LE    |    RP    |
| ------------------ |----------|----------|----------|
| LPR with ResNet101 |  72.72%  |  73.35%  |  52.40%  |


## References

@inproceedings{zhang2021efficient,
  title={Efficient License Plate Recognition via Holistic Position Attention},
  author={Zhang, Yesheng and Wang, Zilei and Zhuang, Jiafan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}

@inproceedings{AOLP2013,
  title={Application-Oriented License Plate Recognition},
  author={Hsu, G.S.; Chen, J.C.; Chung, Y.Z.},
  booktitle={Vehicular Technology, IEEE Transactions on , vol.62, no.2},
  pages={552-561},
  year={2013}
}

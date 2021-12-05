

>📋  A template README.md for code accompanying a Machine Learning paper

# CISC867Project
This is a re-implementation of VIM-LPR for paper "Efficient License Plate Recognition via Holistic Position Attention"(AAAI2021) using tensorflow.
This code is based on jfzhuang's VIM-LPR


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data Preparation and Preprocessing
Download the [AOLP](http://aolpr.ntust.edu.tw/lab/) datasets.<br>
The plates char need to be converted to pure number format.<br>

To extract just the license plate from the AOLP images, run: python plate_extractor1.py or python plate_extractor2.py 
To obtain .npy files which contain the label data and image data in pure number format to use for training and testing, run python prepare_npy_datafile_for_AOLP_AC.py, python prepare_npy_datafile_for_AOLP_LE.py, and python prepare_npy_datafile_for_AOLP_RP.py. This will save the label and image .npy files in the data directory as shown below:
<pre>
├── data  
│     ├── AOLP_AC_plate_images.npy  
│     ├── AOLP_AC_plate_labels_in_number.npy
│     ├── AOLP_LE_plate_images.npy  
│     ├── AOLP_LE_plate_labels_in_number.npy  
│     ├── AOLP_RP_plate_images.npy  
│     ├── AOLP_RP_plate_labels_in_number.npy  

</pre>


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

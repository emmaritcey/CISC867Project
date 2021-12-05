import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Input, multiply, UpSampling2D, Concatenate
from tensorflow.keras.applications.resnet import ResNet101,preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.applications.resnet import ResNet101
import matplotlib.pyplot as plt
import os
import numpy as np


image = load_img('dog.jpg', target_size=(224, 224))

image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)

inputs = Input(shape=(224, 224, 3))

resnet_model = ResNet101(weights='imagenet',input_tensor=inputs, include_top=False)

#resnet_model.summary()

def conv_bn_relu(input, filters, kernel_size=(3, 3), strides=(2, 2), padding='same'):
  x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input) 
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  return x

"""## spatial path"""

#extracts feature maps that are 1/8 of the original image
def spatial_path(input):
  x = conv_bn_relu(input, 64)
  x = conv_bn_relu(x, 128)
  x = conv_bn_relu(x, 256)
  return x

"""## context path"""

def attention_refinement_module(out_channels, input):
  x = layers.GlobalAveragePooling2D(keepdims=True)(input)

  x = layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=(1, 1))(x)  

  x = layers.BatchNormalization()(x)
  x = layers.Activation('sigmoid')(x)
  
  x = layers.Multiply()([input, x])

  return x

def feature_fusion_module(num_classes, input1, input2):
  x = layers.Concatenate(axis=-1)([input1,input2])

  feature = conv_bn_relu(x, num_classes, 3, 1)

  x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(feature)

  x = conv_bn_relu(x, num_classes, 1, 2)
  x = layers.ReLU()(x)

  x = conv_bn_relu(x, num_classes, 1, 2)
  x = sigmoid(x)

  x = layers.Multiply()([feature, x])

  x = layers.Add()([feature, x]) 

  return x

def context_path(input):
  features_list = [layer.output for layer in resnet_model.layers]
  activations_model = tf.keras.Model(inputs=resnet_model.input, outputs=features_list)
  activations = activations_model.predict(image)

  layer_names = []
  for layer in resnet_model.layers:
      layer_names.append(layer.name)
  for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
      if(layer_name == 'conv4_block23_out'):
        feature16 = layer_activation
      if(layer_name == 'conv5_block3_out'):
        feature32 = layer_activation

  tail = tf.math.reduce_mean(feature32, axis=1, keepdims=True)
  tail = tf.math.reduce_mean(tail, axis=2, keepdims=True)

  return feature16, feature32, tail

sp = spatial_path(image)
print("sp.shape: ", sp.shape)

cx1, cx2, tail = context_path(image)

print("cx1.shape = ", cx1.shape)
print("cx2.shape = ", cx2.shape)
print("tail.shape = ", tail.shape)

cx1 = attention_refinement_module(1024, cx1)
cx2 = attention_refinement_module(2048,cx2)

print("cx1.shape = ",cx1.shape)
print("cx2.shape = ",cx2.shape)

cx2 = multiply([tail,cx2])
print("cx2.shape = ", cx2.shape)

# upsampling
cx1 = UpSampling2D(size=2, data_format='channels_last', interpolation='bilinear')(cx1)
cx2 = UpSampling2D(size=4, data_format='channels_last', interpolation='bilinear')(cx2)
print(cx1.shape)
print(cx2.shape)

cx = Concatenate(axis=-1)([cx2, cx1])
print(cx.shape)

#result = feature_fusion_module(sp, cx)       
num_classes  =  12     
result = feature_fusion_module(num_classes, sp, cx)

print("result.shape = ",result.shape)

# upsampling
result1 = UpSampling2D(size=8, data_format='channels_last', interpolation='bilinear')(result)
print("result1.shape = ", result1.shape)

#final convolution
result2 = layers.Conv2D(filters=num_classes, kernel_size=1, strides=1)(result1)
print("result2.shape = ", result2.shape)

result3 = tf.squeeze(result2)
print("result3.shape = ",result3.shape)

result4 = tf.argmax(result3,axis=2)
print(result4.shape)

imgplot = plt.imshow(result4)
plt.show()
#the weights were random data and not trained yet, so the output picture is not a meaningful segmentation
#just for data flow checking
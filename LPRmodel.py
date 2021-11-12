#import necessary packages
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.models import Model
from tensorflow.keras import datasets, layers, models
from keras.layers import Conv2D, Input, multiply, UpSampling2D, Concatenate
from keras.applications.resnet import ResNet101,preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.nn import softmax
import os
import numpy as np
import random as rn
from keras import backend as K


def set_env():
    SEED = 1234

    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, 
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), 
        config=session_conf
    )
    K.set_session(sess)


'''
A string of convolution, batch normalization, and ReLU to be performed consecutively
'''
def conv_bn_relu(input, filters, kernel_size=(3, 3), strides=(2, 2), padding='same'):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input) 
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


'''
spatial path extracts feature maps that are 1/8 of the original image
3 sets of convolution, batch norm, and ReLU
'''
def spatial_path(input):
    print('In spatial path')
    x = conv_bn_relu(input, 64)
    x = conv_bn_relu(x, 128)
    x = conv_bn_relu(x, 256)
    return x


'''
each ARM contains global average pooling, 1x1 convolution, batch norm, sigmoid
context path finishes by sending last two layers into ARM
'''
def attention_refinement_module(out_channels, input):
    print('in ARM')
    x = layers.GlobalAveragePooling2D(keepdims=True)(input)

    x = layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=(1, 1))(x)  

    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)
  
    x = layers.Multiply()([input, x])

    return x


'''
fuses spatial and context features
'''
def feature_fusion_module(num_classes, input1, input2):
    print('In FFM')
    x = layers.Concatenate(axis=-1)([input1,input2])

    feature = conv_bn_relu(x, num_classes, 3, 1)

    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(feature)

    x = layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), use_bias=False)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False)(x) 
    x = sigmoid(x)

    x = layers.Multiply()([feature, x])

    x = layers.Add()([feature, x]) 

    return x  


def context_path(image):
    print('In context path')
    inputs = Input(shape=(50, 160, 3))
    resnet_model = ResNet101(weights='imagenet',input_tensor=inputs, include_top=False)
    
    features_list = [layer.output for layer in resnet_model.layers]
    activations_model = tf.keras.Model(inputs=resnet_model.input, outputs=features_list)
    activations = activations_model.predict(image)

    layer_names = []
    for layer in resnet_model.layers:
        layer_names.append(layer.name)
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        if(layer_name == 'conv4_block23_out'):
            cx1 = layer_activation
        if(layer_name == 'conv5_block3_out'):
            cx2 = layer_activation

    tail = tf.math.reduce_mean(cx2, axis=1, keepdims=True)
    tail = tf.math.reduce_mean(tail, axis=2, keepdims=True)
    print('cx1 shape: ', cx1.shape)
    print('cx2 shape: ', cx2.shape)
    #final two layers of context path sent into ARM
    cx1 = attention_refinement_module(1024, cx1)
    cx2 = attention_refinement_module(2048,cx2)
    print('cx1 shape after ARM: ', cx1.shape)
    print('cx2 shape after ARM: ', cx2.shape)
    
    # upsampling to 28x28
    cx1 = UpSampling2D(size=2, data_format='channels_last', interpolation='bilinear')(cx1)
    cx2 = UpSampling2D(size=4, data_format='channels_last', interpolation='bilinear')(cx2)
    print('cx1 shape after upsampling: ', cx1.shape)
    print('cx2 shape after upsampling: ', cx2.shape)
    #combine
    cx2 = multiply([tail,cx2])
    cx = Concatenate(axis=-1)([cx2, cx1])
       
    print('cx final shape: ', cx.shape)
    
    return cx

'''
to see result of bisenet to visualize semantic/position features
'''
def visualize_features(features):
    # upsampling, use for visualization, I don't think the upsampled versions go into shared classifier?
    featuresUP = UpSampling2D(size=8, data_format='channels_last', interpolation='bilinear')(features)
    #print("semantic_featuresUP.shape = ", featuresUP.shape)
    
    result = tf.squeeze(featuresUP)
    result = tf.argmax(result, axis=2)
    plt.imshow(result)
    

def biseNet(image, numClass1, numClass2):
    print('In BiSeNet')
    #manually pad image for spatial path for proper dimensions
    paddings = tf.constant([[0,0],[7, 7], [0, 0],[0,0]])
    padded_image = tf.pad(image, paddings, "CONSTANT")
    #image sent into spatial and context paths
    sp = spatial_path(padded_image) 
    cp = context_path(image)
    
    #get semantic features by calling FFM with 35 classes
    #get position features by calling FFM with 7 classes    
    num_chars  =  numClass1 
    num_positions = numClass2 
    semantic_features = feature_fusion_module(num_chars, sp, cp)
    position_features = feature_fusion_module(num_positions, sp, cp) 
    
    #visualize_features(semantic_features)
    #visualize_features(position_features)
    
    return semantic_features, position_features
    
'''
final step, performs classification of characters
don't know filters, stride values yet
'''
def sharedClassifier(inputs):

    #Shared Classifier:
    #5x5 convolution, batch norm, ReLU (verify parameter values later)
    #input the result from multiplying semantic_output and pos_att_map
    x = conv_bn_relu(input=inputs, filters=5, kernel_size=(5,5), strides=(1,1), padding='same')(inputs)
    #global pooling
    x = layers.GlobalAveragePooling2D(keepdims=False)(x)
    #fully connected layer for classification
    output = layers.Dense(35)(x)
    
    prediction_indices = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64, name=None)
    
    return prediction_indices

def main():
    
    set_env()
    
    #load in random 224x224 image for now
    image = load_img('dog.jpeg')
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the model
    image = preprocess_input(image)

    
    num_chars = 35
    num_positions = 7
    sem_features, pos_features = biseNet(image, num_chars, num_positions)
    print('semantic features shape: ', sem_features.shape)
    print('position features shape: ', pos_features.shape)
    
    #final convolution
    #sem_features = layers.Conv2D(filters=num_classes, kernel_size=1, strides=1)(sem_features)
    #pos_features = layers.Conv2D(filters=num_positions, kernel_size=1, strides=1)(pos_features)
    
    #apply softmax to position attention maps and batchNorm to semantic features
    pos_att_map = softmax(pos_features)
    semantic_output = layers.BatchNormalization()(sem_features)
    

    #element-wise multiplication, position attention map of each character is used to 
    # modulate the semantic features separately
    #remove background layer from pos_att_map prior to multiplication (I think this is the last layer?)
    product = np.zeros((7,35,28,28))
    for idx in range(0,num_positions):
      #multiply ith slice of pos_att_map by some part of semantic_output tensor
      for jdx in range(0, num_classes):
        layer = tf.math.multiply(pos_att_map[:,:,:,idx], semantic_output[:,:,:,jdx])
        product[idx,jdx,:,:] = layer
        
    product = tf.convert_to_tensor(product)

        
        
    prediction_indices = sharedClassifier(final_inputs)
    ###NEED TO CONVERT THE INDICES TO CHARACTERS/NUMBERS

main()

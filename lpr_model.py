import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101, ResNet50
from tensorflow.keras.layers import Conv2D,Input,GlobalAveragePooling2D,Dense,UpSampling2D,Dropout
from tensorflow.keras.layers import ReLU,Concatenate,BatchNormalization,Multiply,Add,Activation,Softmax
from tensorflow.keras.activations import sigmoid


num_classes  =  35 #total number of potential characters and numbers
num_char = 7       #number of positions/characters in license plates
adNum = 34

#dimensions of images
w0 = 160
h0 = 64
c0 = 3
BatchSize = 24

#base model is either '18', '34', '50', or '101
base_model = '34'

inputs = Input(batch_shape=(BatchSize, h0, w0, c0))


def conv_bn_relu(input, filters, kernel_size=3, strides=2, padding='same'):
    '''
    2d convolution followed by dropout, batch norm, and relu performed many times throughout model
    '''
    x = Conv2D(filters=filters, 
                    kernel_size=kernel_size, 
                    strides=strides, 
                    padding=padding, 
                    use_bias=False)(input) 
     
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x




def spatial_path(inputs):
    '''
    extracts feature maps that are 1/8 of the original image
    '''
    x = conv_bn_relu(inputs, 64)  
    x = conv_bn_relu(x, 128) 
    x = conv_bn_relu(x, 256)
    
    return x



"""
context path
"""

def attention_refinement_module(out_channels, inputs):
    #each ARM contains global average pooling, 1x1 convolution, batch norm, sigmoid
    x = GlobalAveragePooling2D(keepdims=True)(inputs)
    x = Conv2D(filters=out_channels, kernel_size=1)(x)    
    x = Dropout(0.5)(x)    
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Multiply()([inputs, x])    
    return x
  

def feature_fusion_module(num_classes, input1, input2):
    x = Concatenate(axis=-1)([input1,input2]) 
    x = conv_bn_relu(x, filters=512, strides=1)     
    feature = conv_bn_relu(x, filters=num_classes, strides=1)    
    x = GlobalAveragePooling2D(keepdims=True)(feature)
    x = Conv2D(filters=num_classes, kernel_size=1, activation='relu')(x)        
    x = Dropout(0.5)(x)    
    x = Conv2D(filters=num_classes, kernel_size=1, activation='sigmoid')(x)    
    x = Dropout(0.5)(x)   
    x = Multiply()([feature, x])
    x = Add()([feature, x])
    return x
    
  
def feature_fusion_module_pos(num_char, input1, input2):
    x = Concatenate(axis=-1)([input1,input2])  
    feature = conv_bn_relu(x, filters=num_classes, strides=1)
    x = GlobalAveragePooling2D(keepdims=True)(feature)
    x = Conv2D(filters=num_classes, kernel_size=1, activation='relu')(x)                              
    x = Dropout(0.5)(x)   
    x = Conv2D(filters=num_classes, kernel_size=1, activation='sigmoid')(x) 
    x = Dropout(0.5)(x)   
    x = Multiply()([feature, x])
    x = Add()([feature, x])
    return x    
  
     
 
def context_path(inputs, base_model):
    features_list = [layer.output for layer in resnet_model.layers]
    activations_model = Model(inputs=resnet_model.input, outputs=features_list)
    activations = activations_model(inputs)
    
    layer_names = []
    for layer in resnet_model.layers:
        layer_names.append(layer.name)
        
    # get final two layers of base model
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        if base_model == '101':
            if(layer_name == 'conv4_block23_out'):
                feature16 = layer_activation
            if(layer_name == 'conv5_block3_out'):
                feature32 = layer_activation
        if base_model == '50':
            if(layer_name == 'conv4_block6_out'):
                feature16 = layer_activation
            if(layer_name == 'conv5_block3_out'):
                feature32 = layer_activation
        if base_model == '34':
            if (layer_name == 'stage3_unit6_conv2'):
                feature16 = layer_activation
            if (layer_name == 'stage4_unit3_conv2'):
                feature32 = layer_activation
        if base_model == '18':
            if (layer_name == 'stage3_unit2_conv2'):
                feature16 = layer_activation
            if (layer_name == 'stage4_unit2_conv2'):
                feature32 = layer_activation
       
    tail = GlobalAveragePooling2D(keepdims=True)(feature32)

    return feature16, feature32, tail


def shared_classifier(seg_pos):
    x = conv_bn_relu(seg_pos, filters=1000, kernel_size=5, strides=4, padding='same')    
    cls_feat = GlobalAveragePooling2D(keepdims=True)(x)
    cls_feat = tf.squeeze(cls_feat, axis=1)
    cls_feat = tf.squeeze(cls_feat, axis=1)    
    y_one_char = Dense(adNum, input_shape=(1000,), activation='softmax')(cls_feat)      
    return y_one_char
              
        
def bisenet(inputs):  
    # output of spatial path
    sp = spatial_path(inputs)
     
    # output of context path  
    cx1, cx2, tail = context_path(inputs, base_model) 
    
    #send cx1 and cx2 onto ARM, first input is output size which depends on base model used
    if base_model == '101' or base_model =='50':
        cx1 = attention_refinement_module(1024, cx1)
        cx2 = attention_refinement_module(2048, cx2)
    if base_model == '18' or base_model == '34':
        cx1 = attention_refinement_module(256, cx1)
        cx2 = attention_refinement_module(512, cx2)      
    cx2 = Multiply()([tail,cx2])

    # upsampling to appropriate dimensions
    cx1 = UpSampling2D(size=2, data_format='channels_last', interpolation='bilinear')(cx1)
    cx2 = UpSampling2D(size=4, data_format='channels_last', interpolation='bilinear')(cx2)

    cx = Concatenate(axis=-1)([cx2, cx1])
    
    # output of feature fusion module    
    # segmap is semantic segmentation -> predicting actual character
    # posmap is position segmentation -> predicting position of characters
    segmap = feature_fusion_module(num_classes, sp, cx)
    posmap = feature_fusion_module_pos(num_char, sp, cx)
    
    segmapp = Conv2D(filters=num_classes, kernel_size=1, strides=1)(segmap)
    segmapp = Dropout(0.5)(segmapp)
    
    posmapp = Conv2D(filters=num_char, kernel_size=1, strides=1)(posmap)
    posmapp = Dropout(0.5)(posmapp)
    
    segmapp = BatchNormalization()(segmapp)  
    posmapp = Softmax()(posmapp)  

    len1 = posmapp.shape[3] - 1
    
    y_6_char = []
    for i in range(len1): 
        #position attention maps for each char[i]
        posmapp1 = posmapp[:,:,:,i:i+1]
        segmapp1 = segmapp[:,:,:,0:segmapp.shape[3]-1]
             
        seg_pos = Multiply()([segmapp1, posmapp1])
        
        y_one_char = shared_classifier(seg_pos)      
        y_6_char.append(y_one_char)                                 
        
    return [inputs, y_6_char]


inputs = Input(batch_shape=(BatchSize, h0, w0, c0))

#get appropriate pretrained ResNet model
if base_model == '101':
    resnet_model = ResNet101(weights='imagenet',input_tensor=inputs, include_top=False)
if base_model == '50':
    resnet_model = ResNet50(weights='imagenet',input_tensor=inputs, include_top=False)
if base_model == '34':
    from classification_models.keras import Classifiers #no keras ResNet34, import from another library
    ResNet34, preprocess_input = Classifiers.get('resnet34')
    resnet_model = ResNet34(input_shape=(h0,w0,c0), classes=1000)
if base_model == '18':
    from classification_models.keras import Classifiers #no keras ResNet18, import from another library
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    resnet_model = ResNet18(input_shape=(h0,w0,c0), classes=1000)
    
# Freeze all the layers of resnet
for layer in resnet_model.layers[:]:
    layer.trainable = False

#calls model, yx are predictions of each character
inputs, [y0,y1,y2,y3,y4,y5] = bisenet(inputs)
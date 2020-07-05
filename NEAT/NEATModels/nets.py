from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras as K
from keras import regularizers
from keras.layers import BatchNormalization, Activation
from keras.layers import Conv2D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import layers
from keras import models
from keras.layers.core import Lambda
from keras.layers import  TimeDistributed
from keras.layers.merge import concatenate, add



reg_weight = 1.e-4
Kernel_shallow_big_size = 7
Kernel_size = 3
Pool_size = 2
Big_Pool_size = 3
startfilter = 48
Layer1 = [48,32]
Layer2 = [128,64]
Layer3 = [256,128]
Layer4 = [512,256]
Layer5 = [1024,512]

filterA = int(512) 
filterB = int(128) 
"""
Using RESNET and stacked layer style architechtures to define NEAT architecture
"""



class Concat(layers.Layer):

     def __init__(self, axis = -1, name = 'Concat', **kwargs):

          self.axis = axis
          super(Concat, self).__init__(name = name)


 
     def call(self, x):

        y = Lambda(lambda x:layers.concatenate([x[0], x[1]], self.axis))(x)

        return y 

    

    
def TDresnet_v2(input_shape, categories,unit, box_vector,depth = 29,cardinality = 1,  input_weights = None):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
     depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if stage == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = TDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = TDresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             kernel_size=3,

                             conv_first=False)
            y = TDresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = TDresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
            #if res_block%3 == 0:
              #x = TimeDistributed(layers.MaxPooling2D((2,2)))(x)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstm')(x)

    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    
    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model

def TDNONresnet_v2(input_shape, categories,unit, box_vector,depth = 29,cardinality = 1,  input_weights = None):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
     depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if stage == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            
            x = TDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False,
                             strides = strides,
                             activation=activation,
                             batch_normalization=batch_normalization)
            
              
            #if res_block%3 == 0:
              #x = TimeDistributed(layers.MaxPooling2D((2,2)))(x)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstm')(x)

    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    
    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model



def LSTMresnet_v2(input_shape, categories,unit, box_vector,depth = 29,cardinality = 1,  input_weights = None):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
     depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = TDLSTMresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)


                    
                    
                        # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample
                    

            # bottleneck residual unit
            y = TDLSTMresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = TDLSTMresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False)
            y = TDLSTMresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = TDLSTMresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
            #if res_block%3 == 0:
              #x = TimeDistributed(layers.MaxPooling2D((2,2)))(x)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(filters = unit, kernel_size = (3, 3),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstm')(x)

    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
     
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    
    
    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model

def TDresnet_layer(inputs,
                 num_filters=64,
                 kernel_size= Kernel_size,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    
    conv = TimeDistributed(Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4)))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = TimeDistributed(BatchNormalization())(x)
        if activation is not None:
            x = TimeDistributed(Activation(activation))(x)
    else:
        if batch_normalization:
            x = TimeDistributed(BatchNormalization())(x)
        if activation is not None:
            x = TimeDistributed(Activation(activation))(x)
        x = conv(x)
    return x

def TDresnext_layer(inputs,
                 num_filters=64,
                 kernel_size= Kernel_size,
                 strides=1,
                 cardinality = 1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    group_list = []
    grouped_channels = int(num_filters / cardinality)
    
    if cardinality == 1:
        
                x = TimeDistributed(Conv2D(grouped_channels,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4)))(inputs)
                   
                x = TimeDistributed(BatchNormalization())(x)
                if activation is not None:
                    x = TimeDistributed(Activation(activation))(x)
                return x    
    else:
      for c in range(cardinality):
        
          x = Lambda(lambda z: z[:, :, :,:, c * grouped_channels: (c + 1) * grouped_channels])(inputs)

          x = TimeDistributed(Conv2D(grouped_channels,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4)))(x)

          group_list.append(x)
       
      group_merge = concatenate(group_list, axis=-1)
   
      x = TimeDistributed(BatchNormalization())(group_merge)
      if activation is not None:
          x = TimeDistributed(Activation(activation))(x)

    return x
   
def TDLSTMresnet_layer(inputs,
                 num_filters=64,
                 kernel_size= Kernel_size,
                 strides=1,
                 cardinality = 1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = (ConvLSTM2D(num_filters,
                  kernel_size=kernel_size,  activation='relu', strides = strides,
                                  data_format = 'channels_last', return_sequences = True, padding = "same",
                 ))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = TimeDistributed(BatchNormalization())(x)
        if activation is not None:
            x = TimeDistributed(Activation(activation))(x)
    else:
        if batch_normalization:
            x = TimeDistributed(BatchNormalization())(x)
        if activation is not None:
            x = TimeDistributed(Activation(activation))(x)
        x = conv(x)
    return x    

def ThreeDresnet_v2(input_shape, categories,unit, box_vector,depth = 29,cardinality = 1,  input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = ThreeDresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
            #if res_block%3 == 0:
              #x = (layers.MaxPooling3D((1,2,2)))(x)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstm')(x)

    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)

    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model

def ThreeDNonresnet_v2(input_shape, categories,unit, box_vector,depth = 29,cardinality = 1,  input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
      
            x = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False,
                             strides = strides,
                             activation=activation,
                             batch_normalization=batch_normalization)
         

              
            #if res_block%3 == 0:
              #x = (layers.MaxPooling3D((1,2,2)))(x)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstm')(x)

    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)

    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model



def ONETresnet_v2(input_shape, categories,unit, box_vector,depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = ThreeDresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            yz = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_in_TD,
                               kernel_size=3,

                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                z = TDresnet_layer(inputs=z,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            z = K.layers.add([z, yz])
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstmdeep')(branchAdd)



    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name =True)
    
    return model



def ONETNonresnet_v2(input_shape, categories,unit, box_vector,depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
  
            x = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization)


              
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
      
            z = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                               kernel_size=3,

                             conv_first=False,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization)


              
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstmdeep')(branchAdd)



    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name =True)
    
    return model


def OSNET(input_shape, categories,unit, box_vector,depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
  
            x = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization)


              
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
      
            z = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                               kernel_size=3,

                             conv_first=False,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization)


              
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstmdeep')(branchAdd)



    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name =True)
    
    return model


def ORNET(input_shape, categories,unit, box_vector,depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = ThreeDresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            yz = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_in_TD,
                               kernel_size=3,

                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                z = TDresnet_layer(inputs=z,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            z = K.layers.add([z, yz])
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstmdeep')(branchAdd)



    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name =True)
    
    return model




def ONETresnext_v2(input_shape, categories,unit, box_vector,depth = 38,cardinality = 1,  input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                   
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
                   
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = ThreeDresnext_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             cardinality = 1,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = ThreeDresnext_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             cardinality = cardinality,
                             conv_first=False)
            y = ThreeDresnext_layer(inputs=y,
                             num_filters=num_filters_out,
                             cardinality = 1,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = ThreeDresnext_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 cardinality = 1,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            yz = TDresnext_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                             kernel_size=1,
                             strides=strides,
                             cardinality = 1,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            yz = TDresnext_layer(inputs=yz,
                             num_filters=num_filters_in_TD,
                               kernel_size=3,

                             cardinality = cardinality,
                             conv_first=False)
            yz = TDresnext_layer(inputs=yz,
                             num_filters=num_filters_out,
                             cardinality = 1,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                z = TDresnext_layer(inputs=z,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 cardinality = 1,
                                 activation=None,
                                 batch_normalization=False)
              
            z = K.layers.add([z, yz])
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    
    x = ConvLSTM2D(filters = unit, kernel_size = (Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstmdeep')(branchAdd)
    
    
    
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

      model.load_weights(input_weights, by_name =True)
    
    return model



def CNNresnet_v2(input_shape, categories,unit, box_vector,depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = ThreeDresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            yz = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_in_TD,
                               kernel_size=3,

                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                z = TDresnet_layer(inputs=z,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            z = K.layers.add([z, yz])
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    
   
    
    x = Conv3D(filters = unit, kernel_size = (input_shape[0], Kernel_size, Kernel_size),  activation='relu', data_format = 'channels_last',  padding = "same", name = 'cnndeep')(branchAdd)
    x = Lambda(lambda x:x[:,0,:,:,:])(x)     
    
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
            
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model



    
def ThreeDresnet_layer(inputs,
                 num_filters=64,
                 kernel_size=Kernel_size,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=(1,strides,strides),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = (BatchNormalization())(x)
        if activation is not None:
            x = (Activation(activation))(x)
    else:
        if batch_normalization:
            x = (BatchNormalization())(x)
        if activation is not None:
            x = (Activation(activation))(x)
        x = conv(x)
    return x
    
def ThreeDresnext_layer(inputs,
                 num_filters=64,
                 kernel_size=Kernel_size,
                 strides=1,
                 cardinality = 1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    group_list = []
    grouped_channels = int(num_filters / cardinality)
    if cardinality == 1:
                x = Conv3D(grouped_channels,
                  kernel_size=kernel_size,
                  strides=(1,strides,strides),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))(inputs)
                x = (BatchNormalization())(x)
                if activation is not None:
                     x = (Activation(activation))(x)
                return x     
    else:    
      for c in range(cardinality):
        
          x = Lambda(lambda z:z[:, :, :, :,c * grouped_channels: (c + 1) * grouped_channels])(inputs)

          x = Conv3D(grouped_channels,
                  kernel_size=kernel_size,
                  strides=(1,strides,strides),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))(x)

          group_list.append(x)
       
      group_merge = concatenate(group_list, axis=-1)
   
      x = (BatchNormalization())(group_merge)
      if activation is not None:
          x = (Activation(activation))(x)

    return x    
def Timedistributedidentity_block(input_tensor, kernel_size, filters):

    filters1, filters2, filters3 = filters


    x = TimeDistributed(Conv2D(filters1, (1, 1), activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(input_tensor)
    x = TimeDistributed(BatchNormalization())(x)


    x = TimeDistributed(Conv2D(filters2, kernel_size, activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)


    x = TimeDistributed(Conv2D(filters3, (1, 1), activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = layers.add([x, input_tensor])




    return x



def Timedistributedconv_block(input_tensor,
               kernel_size,

               filters, s = 1):

    filters1, filters2, filters3 = filters


    x = TimeDistributed(Conv2D(filters1, (1,1), strides = (s, s), activation='relu' ,kernel_regularizer=regularizers.l2(reg_weight), padding = "same" ))(input_tensor)


    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size,activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)


    x = TimeDistributed(BatchNormalization())(x)


    x = TimeDistributed(Conv2D(filters3, (1,1),activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)

    x = TimeDistributed(BatchNormalization())(x)

    xshort = TimeDistributed(Conv2D(filters3, (1,1), strides = (s,s),activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(input_tensor)

    xshort = TimeDistributed(BatchNormalization())(xshort)

    x = layers.add([x, xshort])



    return x


def ThreeDidentity_block(input_tensor, kernel_size, filters):

    filters1, filters2, filters3 = filters


    x = (Conv3D(filters1, (1, 1, 1), activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(input_tensor)
    x = (BatchNormalization())(x)


    x = (Conv3D(filters2, kernel_size, activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)
    x = (BatchNormalization())(x)


    x = (Conv3D(filters3, (1, 1, 1), activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)
    x = (BatchNormalization())(x)

    x = layers.add([x, input_tensor])




    return x



def ThreeDconv_block(input_tensor,
               kernel_size,

               filters, s = 1):

    filters1, filters2, filters3 = filters


    sT = 1
    x = (Conv3D(filters1, (1,1,1), strides = (sT, s, s), activation='relu' ,kernel_regularizer=regularizers.l2(reg_weight), padding = "same" ))(input_tensor)


    x = (BatchNormalization())(x)

    x = (Conv3D(filters2, kernel_size,activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)


    x = (BatchNormalization())(x)


    x = (Conv3D(filters3, (1,1,1),activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)

    x = (BatchNormalization())(x)
   
        
    
    xshort = (Conv3D(filters3, (1, 1,1), strides = (sT, s,s),activation='relu',kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(input_tensor)

    xshort = (BatchNormalization())(xshort)

    x = layers.add([x, xshort])



    return x
#### Combo network combining the capabilities of 3D CNN and LSTM ####
 
def ResidualBlocks(x, input_shape,categories):



    x = TimeDistributed(Conv2D(Layer1[0], (Kernel_shallow_big_size,Kernel_shallow_big_size),activation='relu' ,kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)


    x = Timedistributedconv_block(x,Kernel_small_size, (Layer1[0],Layer1[1],Layer1[1]))
    x = Timedistributedidentity_block(x,Kernel_small_size,(Layer1[0],Layer1[1],Layer1[1]))
    x = TimeDistributed(layers.MaxPooling2D((Pool_size,Pool_size)))(x)
   

    x = Timedistributedconv_block(x,Kernel_small_size, (Layer2[0],Layer2[1],Layer2[1]))
    x = Timedistributedidentity_block(x,Kernel_small_size,(Layer2[0],Layer2[1],Layer2[1]))
    


    x = Timedistributedconv_block(x,Kernel_small_size, (Layer3[0],Layer3[1],Layer3[1]))
    x = Timedistributedidentity_block(x,Kernel_small_size,(Layer3[0],Layer3[1],Layer3[1]))
    x = TimeDistributed(layers.MaxPooling2D((Pool_size,Pool_size)))(x)
    
    x = Timedistributedconv_block(x,Kernel_small_size, (Layer4[0],Layer4[1],Layer4[1]))
    x = Timedistributedidentity_block(x,Kernel_small_size,(Layer4[0],Layer4[1],Layer4[1]))
    x = TimeDistributed(layers.MaxPooling2D((Pool_size,Pool_size)))(x)

    return x       
def ONET(input_shape, categories,unit, box_vector,depth = 29,  input_weights = None):
    
    
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    
    branchRes = ResidualBlocks(img_input, input_shape, categories)
    branchCNN = ResidualBlocks3D(img_input, input_shape, categories)
    branchAdd = layers.add([branchRes,branchCNN])
    
    
    
    x = ConvLSTM2D(filters = unit, kernel_size = (3, 3),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = 'lstmdeep')(branchAdd)
    
    
    
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
    output_cat = (Conv2D(categories, (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'deep1'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[1]/4),round(input_shape[2]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'deep2'))(input_box)
    
    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
  
    
  

def ResidualBlocks3D(x, input_shape,categories):

    
    
    Pool_sizeT = 1
        
 

    x = (Conv3D(Layer1[0], (Kernel_shallow_big_size,Kernel_shallow_big_size,Kernel_shallow_big_size),activation='relu' ,kernel_regularizer=regularizers.l2(reg_weight),  padding="same"))(x)
    x = (BatchNormalization())(x)


    x = ThreeDconv_block(x,Kernel_small_size, (Layer1[0],Layer1[1],Layer1[1]))
    x = ThreeDidentity_block(x,Kernel_small_size,(Layer1[0],Layer1[1],Layer1[1]))
    x = (layers.MaxPooling3D((Pool_sizeT,Pool_size,Pool_size)))(x)
    

    x = ThreeDconv_block(x,Kernel_small_size, (Layer2[0],Layer2[1],Layer2[1]))
    x = ThreeDidentity_block(x,Kernel_small_size,(Layer2[0],Layer2[1],Layer2[1]))
    
    


    x = ThreeDconv_block(x,Kernel_small_size, (Layer3[0],Layer3[1],Layer3[1]))
    x = ThreeDidentity_block(x,Kernel_small_size,(Layer3[0],Layer3[1],Layer3[1]))
    x = (layers.MaxPooling3D((Pool_sizeT,Pool_size,Pool_size)))(x)
   
    

        


    x = ThreeDconv_block(x,Kernel_small_size, (Layer4[0],Layer4[1],Layer4[1]))
    x = ThreeDidentity_block(x,Kernel_small_size,(Layer4[0],Layer4[1],Layer4[1]))
    x = (layers.MaxPooling3D((Pool_sizeT,Pool_size,Pool_size)))(x)

    return x



def resnet_v2(input_shape, categories, box_vector, depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = layers.Input(shape = (None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
      
    

    output_cat = (Conv2D(categories, (round(input_shape[0]/4),round(input_shape[1]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[0]/4),round(input_shape[1]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])

    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
  
def Nonresnet_v2(input_shape, categories, box_vector, depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = layers.Input(shape = (None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit

            x = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             conv_first=False,
                             strides = strides,
                             activation=activation,
                             batch_normalization=batch_normalization)
           
 
              
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
      
    

    output_cat = (Conv2D(categories, (round(input_shape[0]/4),round(input_shape[1]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[0]/4),round(input_shape[1]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])

    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
      


def resnext_v2(input_shape, categories, box_vector, depth = 38, cardinality = 1, input_weights = None):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = layers.Input(shape = (None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
                   
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnext_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             cardinality = 1,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnext_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size=3,

                             cardinality = cardinality,
                             conv_first=False)
            y = resnext_layer(inputs=y,
                             num_filters=num_filters_out,
                             cardinality = 1,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnext_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 cardinality = 1,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
 

    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

    output_cat = (Conv2D(categories, (round(input_shape[0]/4),round(input_shape[1]/4)),activation= 'softmax' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D((box_vector), (round(input_shape[0]/4),round(input_shape[1]/4)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)


    block = Concat(-1)
    outputs = block([output_cat,output_box])

    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)

    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
    
   
def resnet_layer(inputs,
                 num_filters=64,
                 kernel_size=Kernel_size,
                 strides=1,
                 cardinality = 1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
    
def resnext_layer(inputs,
                 num_filters=64,
                 kernel_size= Kernel_size,
                 strides=1,
                 cardinality = 1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    group_list = []
   
    grouped_channels = int(num_filters / cardinality)
    if cardinality == 1:
        
        x = (Conv2D(grouped_channels,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4)))(inputs)
        x = (BatchNormalization())(x)
        if activation is not None:
              x = (Activation(activation))(x)
        return x 

         
    else:    
      for c in range(cardinality):
        
          x = Lambda(lambda z:z[:, :, :, c * grouped_channels: (c + 1) * grouped_channels])(inputs)

          x = (Conv2D(grouped_channels,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4)))(x)

          group_list.append(x)
        
      group_merge = concatenate(group_list, axis=-1)
   
    
    
      x = (BatchNormalization())(group_merge)
      if activation is not None:
          x = (Activation(activation))(x)

      return x
   

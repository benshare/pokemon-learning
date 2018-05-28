import numpy as np
import tensorflow as tf
from tflearn.layers.core import input_data, fully_connected#, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn import data_augmentation

class ShallowNet():
    
    def __init__(self, num_layers=2, image_width=128, filter_sizes=[5, 5],\
             nums_of_filters=[32, 64], num_channels=4, num_classes=19):
        self.params = [num_layers, image_width, filter_sizes, nums_of_filters, num_channels, num_classes]
    
    def tflearn_model(self):
        num_layers, image_width, filter_sizes, nums_of_filters, num_channels, num_classes = self.params
        
        input_shape = (None, image_width, image_width, num_channels)
        
        if num_layers > len(filter_sizes):
            filter_sizes.extend([5 for i in range(num_layers - len(filter_sizes))])
        if num_layers > len(nums_of_filters):
            nums_of_filters.extend(\
                [5 for i in range(num_layers - len(nums_of_filters))])

        img_aug = data_augmentation.ImageAugmentation()
        img_aug.add_random_flip_leftright()
#         img_aug.add_random_rotation()
#         img_aug.add_random_crop(input_shape, 12)
        
        net = input_data(shape=input_shape, data_augmentation=img_aug, name='input')
        for layer_num in range(num_layers):
            net = conv_2d(net, nums_of_filters[layer_num], filter_sizes[layer_num], activation='relu', name=('convl%d' %num_layers))
            net = max_pool_2d(net, 2, padding='valid')
        net = fully_connected(net, 64, activation='relu', name='fc1')
        net = fully_connected(net, num_classes, name='fc2')
        net = regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
        return net
    
    def model_init_fn(self, inputs, DEBUG=False, beta=0.0):
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
        
        num_layers, image_width, filter_sizes, nums_of_filters, num_channels, num_classes = self.params
        initializer = tf.variance_scaling_initializer(scale=2.0)
        input_shape = (image_width, image_width, num_channels)
        if num_layers > len(filter_sizes):
            filter_sizes.extend([5 for i in range(num_layers - len(filter_sizes))])
        if num_layers > len(nums_of_filters):
            nums_of_filters.extend(\
                [5 for i in range(num_layers - len(nums_of_filters))])

        def conv_section(num):
            conv_shape = (image_width / (2**num), image_width / (2**num), num_channels)
            section = [
                tf.layers.Conv2D(input_shape=conv_shape, filters=(nums_of_filters[num]),\
                     kernel_size=filter_sizes[num], strides=1, padding="same",\
                     activation=tf.nn.relu, kernel_regularizer=regularizer),
                tf.keras.layers.MaxPooling2D(padding="valid")
            ]
            return section

        fc_section = [
            tf.layers.Flatten(input_shape=input_shape),
            tf.layers.Dense(64, kernel_initializer=initializer,\
                activation=tf.nn.relu, kernel_regularizer=regularizer),
            tf.layers.Dense(num_classes, kernel_initializer=initializer, kernel_regularizer=regularizer)
        ]

        layers = []
        for layer_num in range(num_layers):
            layers.extend(conv_section(layer_num))
        layers.extend(fc_section)
        partials = []
        if DEBUG:
            for p in range(1,5):
                partials.append(tf.keras.Sequential(layers[:p])(inputs))
        model = tf.keras.Sequential(layers)
        return partials, model(inputs)

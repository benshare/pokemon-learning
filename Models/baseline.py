import numpy as np
import tensorflow as tf


def shallowNet(num_layers=2, image_width=128, filter_sizes=[5, 5], nums_of_filters=[32, 64], num_classes=19):
    initializer = tf.variance_scaling_initializer(scale=2.0)
    input_shape = (image_width, image_width, 4)
    if num_layers > len(filter_sizes):
        filter_sizes.extend([5 for i in range(num_layers - len(filter_sizes))])
    if num_layers > len(nums_of_filters):
        nums_of_filters.extend([5 for i in range(num_layers - len(nums_of_filters))])

    def conv_section(num):
        conv_shape = (image_width / (2**num), image_width / (2**num), 4)
        section = [
            tf.layers.Conv2D(input_shape=conv_shape, filters=(32 * 2**num), kernel_size=filter_sizes[num],\
                            strides=1, padding="same", activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(padding="valid")
        ]
        return section

    fc_section = [
        tf.layers.Flatten(input_shape=input_shape),
        tf.layers.Dense(64, kernel_initializer=initializer, activation=tf.nn.relu),
        tf.layers.Dense(num_classes, kernel_initializer=initializer)
    ]
            
    layers = []
    for conv_num in range(num_conv_sections):
        layers.extend(conv_section(conv_num))
    layers.extend(fc_section)
    model = tf.keras.Sequential(layers)
    return model(inputs)
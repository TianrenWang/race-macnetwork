# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from MACRNN import *
import tensorflow as tf
from transformer import Transformer

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def MAC_network_generator(d_model, num_classes, max_steps):
    """Generator for CIFAR-10 ResNet v2 models.

    Args:
        d_model: A single integer for the size of the model.
        num_classes: The number of possible classes for image classification.
        max_steps: The maximum number of reasoning steps.

    Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the MAC Network model.

    Raises:
        ValueError: If `resnet_size` is invalid.
    """

    def model(knowledge, question, is_training):
        """Constructs the MAC Network model given the inputs."""

        step_encoder = positional_encoding(max_steps, d_model)

        decoder1 = tf.keras.layers.Dense(768)
        decoder2 = tf.keras.layers.Dense(512)

        decoded_knowledge = tf.nn.tanh(decoder1(knowledge))
        decoded_knowledge = tf.nn.tanh(decoder2(decoded_knowledge))
        decoded_quest = tf.nn.tanh(decoder1(question))
        decoded_quest = tf.nn.tanh(decoder2(decoded_quest))

        context_words = Transformer(2, d_model, 8, d_model)(decoded_quest, decoded_quest, is_training)
        question_repre = tf.reduce_mean(context_words, 1)

        mac_cell = MAC_Cell(d_model, max_steps) #,

        decoded_knowledge = tf.keras.Input((512, 512), tensor=decoded_knowledge)
        context_words = tf.keras.Input((128, 512), tensor=context_words)
        question_repre = tf.keras.Input((512), tensor=question_repre)

        mac_rnn = tf.keras.layers.RNN(mac_cell)
        final_output = mac_rnn(step_encoder, constants=[decoded_knowledge, context_words, question_repre], training=is_training)

        memory = tf.slice(final_output[1], [0, 0, 0], [-1, 1, -1])
        memory = tf.squeeze(memory, axis=1) # getting rid of the iteration axis

        output_cell = Output(d_model, num_classes)
        output = output_cell(memory, question_repre, is_training)

        return output # Softmax classified [batch, classes]

    return model

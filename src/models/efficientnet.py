# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
import math
from typing import List

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file

# from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess

from keras_efficientnets.config import BlockArgs, get_default_block_list
from keras_efficientnets.custom_objects import EfficientNetConvInitializer
from keras_efficientnets.custom_objects import EfficientNetDenseInitializer
from keras_efficientnets.custom_objects import Swish, DropConnect


__all__ = ['EfficientNet',
           'EfficientNetB0',
           'EfficientNetB1',
           'EfficientNetB2',
           'EfficientNetB3',
           'EfficientNetB4',
           'EfficientNetB5',
           'EfficientNetB6',
           'EfficientNetB7',
           'preprocess_input']


import os

import os
import json
import warnings
import numpy as np

from keras.applications import backend
from keras.applications import layers
from keras.applications import models
from keras.applications import utils

_KERAS_BACKEND = backend
_KERAS_LAYERS = layers
_KERAS_MODELS = models
_KERAS_UTILS = utils



WEIGHTS_PATH_NO_TOP = '/home/work/user-job-dir/src/weights/efficientnet-b5_notop.h5'

CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/'
                    'model_zoo/resnet/imagenet_class_index.json')
backend = None
layers = None
models = None
keras_utils = None

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

__version__ = '1.0.7'


def _preprocess_numpy_input(x, data_format, mode, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.

    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed Numpy array.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


def _preprocess_symbolic_input(x, data_format, mode, **kwargs):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: Input tensor, 3D or 4D.
        data_format: Data format of the image tensor.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed tensor.
    """
    global _IMAGENET_MEAN

    backend, _, _, _ = get_submodules_from_kwargs(kwargs)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if backend.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    if _IMAGENET_MEAN is None:
        _IMAGENET_MEAN = backend.constant(-np.array(mean))

    # Zero-center by mean pixel
    if backend.dtype(x) != backend.dtype(_IMAGENET_MEAN):
        x = backend.bias_add(
            x, backend.cast(_IMAGENET_MEAN, backend.dtype(x)),
            data_format=data_format)
    else:
        x = backend.bias_add(x, _IMAGENET_MEAN, data_format)
    if std is not None:
        x /= std
    return x


def preprocess_input(x, data_format=None, mode='caffe', **kwargs):
    """Preprocesses a tensor or Numpy array encoding a batch of images.

    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed tensor or Numpy array.

    # Raises
        ValueError: In case of unknown `data_format` argument.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format,
                                       mode=mode, **kwargs)
    else:
        return _preprocess_symbolic_input(x, data_format=data_format,
                                          mode=mode, **kwargs)


def decode_predictions(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    global CLASS_INDEX

    backend, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = keras_utils.get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH,
            cache_subdir='models',
            file_hash='c2c37ea517e94d9795004a39431a14cb',
            cache_dir=os.path.join(os.path.dirname(__file__), '..'))
        with open(fpath) as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.

    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: In case of invalid argument values.
    """

    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape



# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)
            x = layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block


def EfficientNet(input_shape,
                 block_args_list: List[BlockArgs],
                 width_coefficient: float,
                 depth_coefficient: float,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.,
                 drop_connect_rate=0.,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 depth_divisor=8,
                 min_depth=None,
                 data_format=None,
                 default_size=None,
                 **kwargs):
    """
    Builder model for EfficientNets.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        block_args_list: Optional List of BlockArgs, each
            of which detail the arguments of the MBConvBlock.
            If left as None, it defaults to the blocks
            from the paper.
        width_coefficient: Determines the number of channels
            available per layer. Compound Coefficient that
            needs to be found using grid search on a base
            configuration model.
        depth_coefficient: Determines the number of layers
            available to the model. Compound Coefficient that
            needs to be found using grid search on a base
            configuration model.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        batch_norm_momentum: Float, default batch normalization
            momentum. Obtained from the paper.
        batch_norm_epsilon: Float, default batch normalization
            epsilon. Obtained from the paper.
        depth_divisor: Optional. Used when rounding off the coefficient
             scaled channels and depth of the layers.
        min_depth: Optional. Minimum depth value in order to
            avoid blocks with 0 layers.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.
        default_size: Specifies the default image size of the model

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_default_block_list()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # Determine proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=min_size,
                                      data_format=data_format,
                                      require_flatten=include_top,
                                      weights=weights)

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs
    x = layers.Conv2D(
        filters=round_filters(32, width_coefficient,
                              depth_divisor, min_depth),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)

    # Head part
    x = layers.Conv2D(
        filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    outputs = x

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    from keras.engine.topology import get_source_inputs
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs, outputs)

    # Load weights
    if weights == 'imagenet':
        if default_size == 224:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b0.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b0_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 240:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b1.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b1_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 260:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b2.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b2_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 300:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b3.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b3_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 380:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b4.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b4_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 456:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b5.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5.h5",
                    cache_subdir='models')
            else:
                weights_path = WEIGHTS_PATH_NO_TOP
            model.load_weights(weights_path)

        # TODO: When weights for efficientnet-b6 and efficientnet-b7 becomes available, uncomment this section and update
        #           the ValueError message below (line 537: ValueError('ImageNet weights can only be loaded with EfficientNetB0-5'))
        # elif default_size == 528:
        #     if include_top:
        #         weights_path = get_file(
        #             'efficientnet-b6.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6.h5",
        #             cache_subdir='models')
        #     else:
        #         weights_path = get_file(
        #             'efficientnet-b6_notop.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6_notop.h5",
        #             cache_subdir='models')
        #     model.load_weights(weights_path)
        #
        # elif default_size == 600:
        #     if include_top:
        #         weights_path = get_file(
        #             'efficientnet-b7.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7.h5",
        #             cache_subdir='models')
        #     else:
        #         weights_path = get_file(
        #             'efficientnet-b7_notop.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7_notop.h5",
        #             cache_subdir='models')
        #     model.load_weights(weights_path)

        else:
            raise ValueError('ImageNet weights can only be loaded with EfficientNetB0-5')

    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B0.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=224)


def EfficientNetB1(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B1.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=240)


def EfficientNetB2(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B2.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.1,
                        depth_coefficient=1.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=260)


def EfficientNetB3(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B3.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.2,
                        depth_coefficient=1.4,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=300)


def EfficientNetB4(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B4.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.4,
                        depth_coefficient=1.8,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=380)


def EfficientNetB5(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B5.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.6,
                        depth_coefficient=2.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=456)


def EfficientNetB6(input_shape=None,
                   include_top=True,
                   weights=None,
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B6.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.8,
                        depth_coefficient=2.6,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=528)


def EfficientNetB7(input_shape=None,
                   include_top=True,
                   weights=None,
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B7.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=2.0,
                        depth_coefficient=3.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=600)


if __name__ == '__main__':
    import os
    from keras.models import load_model

    model = EfficientNetB0(include_top=True)
    model.summary()

    model.save("temp.h5")

    if os.path.exists('temp.h5'):
        model = load_model('temp.h5', compile=False)
        model.summary()

    else:
        raise FileNotFoundError("Keras model file not found !")

    if os.path.exists('temp.h5'):
        os.remove('temp.h5')

# -*- coding: utf-8 -*-
import os
import warnings
import multiprocessing
from glob import glob
# from keras.applications.ne
import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam
from keras.layers import Flatten, Dense,GlobalMaxPooling2D,concatenate,Dropout,Input
from keras.callbacks import TensorBoard, Callback
from moxing.framework import file
import keras
from data_gen import data_flow
# from data_gene import data_flow
from models.resnext50 import ResNext50
from models.resnet50 import ResNet50
from models.inception_v3 import InceptionV3
from models.xception import Xception
from models.efficientnet import EfficientNetB5
from models.densenet import DenseNet201


backend.set_image_data_format('channels_last')

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
def model_fn(FLAGS, objective, optimizer, metrics,input_shape=None):
    """
    pre-trained resnet50 model
    """
    # base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(FLAGS.input_size, FLAGS.input_size, 3), pooling=None, classes=FLAGS.num_classes)
    # base_model = ResNext50(weights=None,
    #                       include_top=False,
    #                       pooling=None,
    #                       input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
    #                       classes=FLAGS.num_classes)
    # base_model = ResNet50(weights='imagenet',
    #                        include_top=False,
    #                        pooling=None,
    #                        input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
    #                        classes=FLAGS.num_classes)
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=True,
                                      weights='imagenet')
    # # inp=Input(shape=(FLAGS.batch_size,FLAGS.input_size,FLAGS.input_size,3))
    inp=Input(input_shape)
    base_model=EfficientNetB5(
                            # input_tensor=inp,
                            weights='/home/work/user-job-dir/src/weights/efficientnet-b5_notop.h5',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    # base_model.load_weights('/home/work/user-job-dir/src/weights/efficientnet-b5_notop.h5')
    base_model1 = InceptionV3(
                          input_tensor=inp,
                          weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    base_model2 = Xception(
                          input_tensor=inp,
                          weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    base_model3=ResNet50(
                          input_tensor=inp,
                          weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model1.layers:
        layer.trainable = True
    for layer in base_model2.layers:
        layer.trainable = True
    for layer in base_model3.layers:
        layer.trainable =True
    # x1 = base_model1.output
    # x1 = Flatten()(x1)
    inception = base_model1.output
    xception=base_model2.output
    densenet=base_model3.output
    top1_model=GlobalMaxPooling2D(data_format='channels_last')(inception)
    top2_model = GlobalMaxPooling2D(data_format='channels_last')(xception)
    top3_model = GlobalMaxPooling2D(data_format='channels_last')(densenet)
    top1_model = Flatten()(top1_model)
    top2_model = Flatten()(top2_model)
    top3_model = Flatten()(top3_model)
    t=concatenate([top1_model,top2_model,top3_model],axis=1)
    # # # t=Flatten()(t)
    # top_model=Dense(512,activation='relu')(t)
    # top_model=Dropout(rate=0.5)(top_model)

    for layer in base_model.layers:
        layer.trainable = True
    out=base_model.output
    out=Flatten()(out)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


class LossHistory(Callback):
    def __init__(self, FLAGS):
        super(LossHistory, self).__init__()
        self.FLAGS = FLAGS

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        save_path = os.path.join(self.FLAGS.train_local, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
        self.model.save_weights(save_path)
        if self.FLAGS.train_url.startswith('s3://'):
            save_url = os.path.join(self.FLAGS.train_url, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
            file.copy(save_path, save_url)
        print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path


def train_model(FLAGS):
    # data flow generator
    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)

    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    objective = 'binary_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    if FLAGS.restore_model_path != '' and file.exists(FLAGS.restore_model_path):
        if FLAGS.restore_model_path.startswith('s3://'):
            restore_model_name = FLAGS.restore_model_path.rsplit('/', 1)[1]
            file.copy(FLAGS.restore_model_path, '/cache/tmp/' + restore_model_name)
            model.load_weights('/cache/tmp/' + restore_model_name)
            os.remove('/cache/tmp/' + restore_model_name)
        else:
            model.load_weights(FLAGS.restore_model_path)
    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)
    tensorBoard = TensorBoard(log_dir=FLAGS.train_local)
    history = LossHistory(FLAGS)
    # STEP_SIZE_TRAIN = train_sequence.n

    # STEP_SIZE_VALID = validation_sequence.n
    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorBoard],
        # validation_steps=STEP_SIZE_VALID,
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )
    # count=train_sequence.get_count()
    # for n in range(FLAGS.max_epochs):
    #     for i in range(count):
    #         batch_train_x,batch_train_y = train_sequence.next_batch()
    #         batch_val_x, batch_val_y=validation_sequence.next_batch()
    #         model.fit(x=batch_train_x,
    #               y=batch_train_y,
    #             verbose=1,
    #             callbacks=[history, tensorBoard],
    #             # validation_steps=STEP_SIZE_VALID,
    #             validation_data=(batch_val_x, batch_val_y),
    #             shuffle=True)
    print('training done!')

    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        save_pb_model(FLAGS, model)

    if FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from eval import load_test_data
        img_names, test_data, test_labels = load_test_data(FLAGS)
        predictions = model.predict(test_data, verbose=0)

        right_count = 0
        for index, pred in enumerate(predictions):
            predict_label = np.argmax(pred, axis=0)
            test_label = test_labels[index]
            if predict_label == test_label:
                right_count += 1
        accuracy = right_count / len(img_names)
        print('accuracy: %0.4f' % accuracy)
        metric_file_name = os.path.join(FLAGS.train_local, 'metric.json')
        metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
        with open(metric_file_name, "w") as f:
            f.write(metric_file_content + '\n')
    print('end')

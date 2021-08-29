from re import X
import os
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Flatten, Activation, ZeroPadding2D, Add
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model

from keras import backend as K 

def plain_conv_block(inputs, num_filters = 16, alpha = 1, kernel_size = 2, pooling = None, block_id=1, activation = 'relu'):

    x = Conv2D(int(num_filters*alpha), kernel_size, padding='same', use_bias = False, name='conv_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_%d_bn' % block_id)(x)
    x = Activation(activation, name='conv_%d_act' % block_id)(x)

    if pooling:
        x = MaxPooling2D(pool_size = pooling, name='conv_%d_pool' % block_id)(x)
    return x

def dw_conv_block(inputs, num_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1, activation = 'relu'):
    channel_axis = 1
    pointwise_conv_filters = int(num_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(activation, name='conv_dw_%d_act' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)

    x = Activation(activation, name='conv_pw_%d_act' % block_id)(x)

    return x


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def inverted_res_block(x, num_filters, expansion, kernel_size, stride, activation, block_id):
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = x.shape[3]

    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(_depth(infilters * expansion),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3,
                                momentum=0.999,
                                name=prefix + 'expand/BatchNorm')(x)
        x = Activation(activation, name=prefix + 'expand/Act')(x)

    if stride == 2:
        x = ZeroPadding2D(padding=((0, 1), (0, 1)),
                                 name=prefix + 'depthwise/pad')(x)
        x = DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = Activation(activation, name=prefix + 'depthwise/Act')(x)

    x = Conv2D(num_filters,
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3,
                        momentum=0.999,
                        name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == num_filters:
        x = Add(name=prefix + 'Add')([shortcut, x])
    return x

def plain_Conv2D(x):

    layers = [
        [16, 3, None],
        [16, 2, 2],
        [16, 2, 2],
        [32, 3, 2],
        [128, 2, None]
    ]

    for i, layer in enumerate(layers):
        x = plain_conv_block(x, num_filters = layer[0], alpha = 1, kernel_size = layer[1], pooling = layer[2], block_id=i, activation = 'relu')

    return x

def DW_Conv2D(x):

    layers = [
        [16, 2, 2],
        [16, 2, 1],
        [32, 3, 2],
        [128, 2, 1]
    ]

    x = plain_conv_block(x, num_filters = 16, alpha = 2, kernel_size = 2, pooling = None, block_id=0, activation = 'relu')

    for i, layer in enumerate(layers):
        x = dw_conv_block(x, layer[0], 1, depth_multiplier=1, strides=layer[2], block_id=i, activation = 'relu')

    return x

def res_Conv2D(x):

    layers = [
        [16, 3, None],
        [16, 2, 2],
        [16, 2, 2],
        [32, 3, 2],
        [128, 2, None]
    ]

    for i, layer in enumerate(layers):
        x = inverted_res_block(x, num_filters = layer[0], expansion = 2, kernel_size = layer[1], stride = layer[2], activation = 'relu', block_id=i)

    return x


def get_model(type, n_classes, n_slots, audio_params):

    K.clear_session()

    main_input = Input(shape=(150, audio_params['num_cepstral'], 1), name='main_input')

    x = eval(type)(main_input)

    x = GlobalMaxPooling2D()(x)

    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)

    slot_dense = Dense(n_slots*2)(x)
    slot_reshape = Reshape(target_shape = (2, n_slots))(slot_dense)
    slot_output = Softmax(name='slot_output')(slot_reshape)

    intent_output = Dense(n_classes, activation='softmax', name='intent_output', use_bias = False)(x)

    model = Model(inputs=main_input, outputs=[intent_output, slot_output], name = type)
    return model



def tflite_convert(model, model_path, audio_params):

    if not model:
        model = tf.keras.models.load_model(model_name)

    def representative_dataset():
        for i in range(len(test_data)):
            wav_file = os.path.join(*prefix, test_data['path'][i])
            audio, sample_rate = librosa.load(wav_file, sr=16000, res_type='kaiser_best')
            audio = librosa.util.fix_length(audio, 16000*3)
            features = generate_features(False, audio, SAMPLING_RATE, 
                            WIN_SIZE_MS, WIN_INCREASE_MS, 32, 
                            NUM_CEPSTRAL, MIN_FREQ, MAX_FREQ)
            
            features = features['features']
            X = np.expand_dims(features, axis = -1)
            X = np.expand_dims(X, axis = 0)
            yield [X.astype(np.float32)]

    model.input.set_shape(1 + model.input.shape[1:])
                
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_type = tf.int8
    converter.inference_input_type = tf.int8 
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    # Save the model.
    tflite_filename = os.path.abspath(model_path).split('.')[0] + '.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_quant_model)

    return tflite_filename

def tflite_micro_convert(tflite_filename):
    pass
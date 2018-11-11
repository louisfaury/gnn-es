
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def compile_function(inputs, outputs):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))
    return run


def compile_scipy_optimizer(optimizer, inputs):
    def run(*input_vals):
        session = tf.get_default_session()
        return optimizer.minimize(session, feed_dict=dict(list(zip(inputs, input_vals))))
    return run


def create_model(model_class, input_size=None, model_args=None):
    input = Input(shape=(input_size,))
    if model_args is None:
        model_args = dict()
    output = model_class(input_tensor=input,
                         **model_args).out
    return Model([input], output)


def reshape_vector_to_tensor_list(vector, tensor_list_shapes):
    tensorized_vector =[]
    index = 0
    for shape in tensor_list_shapes:
        size = np.prod(shape)
        partial_vector = vector[index:index+size]
        tensorized_vector.append(np.reshape(partial_vector, shape))
        index += size
    return tensorized_vector


def flatten_tensor_list(tensor_list):
    flat_tensor = tf.concat([tf.reshape(tensor, [-1]) for tensor in tensor_list], axis=0)
    return flat_tensor

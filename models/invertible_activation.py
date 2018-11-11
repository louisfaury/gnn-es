
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.layers import Lambda, Reshape


class InvertibleActivationFunction(ABC):
    @abstractmethod
    def forward(self, input_tensor):
        pass

    @abstractmethod
    def backward(self, input_tensor):
        pass

    @abstractmethod
    def log_det_jacobian(self, input_tensor):
        pass


class InvertibleIdentity(InvertibleActivationFunction):
    def forward(self, input_tensor):
        return input_tensor

    def backward(self, input_tensor):
        return input_tensor

    def log_det_jacobian(self, input_tensor):
        log_det_jacobian = Lambda(lambda x: tf.zeros(shape=(tf.shape(x)[0],)))(input_tensor)
        log_det_jacobian = Reshape((1,))(log_det_jacobian)
        return log_det_jacobian


class InvertibleHyperbolicTangent(InvertibleActivationFunction):
    def forward(self, input_tensor):
        return Lambda(lambda x: tf.tanh(x))(input_tensor)

    def backward(self, input_tensor):
        return Lambda(lambda x: tf.atanh(x))(input_tensor)

    def log_det_jacobian(self, input_tensor):
        log_det_jacobian = Lambda(lambda x: -tf.reduce_sum(tf.log(1 - x ** 2 + 1e-5), axis=1))(input_tensor)
        log_det_jacobian = Reshape((1,))(log_det_jacobian)
        return log_det_jacobian


class InvertibleSigmoid(InvertibleActivationFunction):
    def forward(self, input_tensor):
        return Lambda(lambda x: 1./(1+tf.exp(-x)))(input_tensor)

    def backward(self, input_tensor):
        return Lambda(lambda x: -tf.log(1./x-1))(input_tensor)

    def log_det_jacobian(self, input_tensor):
        log_det_jacobian = Lambda(lambda x: -tf.reduce_sum(tf.log(x-x**2), axis=1))(input_tensor)
        log_det_jacobian = Reshape((1,))(log_det_jacobian)
        return log_det_jacobian


def create_invertible_activation_function(activation_name):
    if activation_name == 'identity':
        return InvertibleIdentity()
    elif activation_name == 'sigmoid':
        return InvertibleSigmoid()
    elif activation_name == 'tanh':
        return InvertibleHyperbolicTangent()
    else:
        raise ValueError('Unknown invertible activation function')

from abc import ABC, abstractmethod
from utils.tensorflow_utils import compile_function
import tensorflow as tf
import logging


class Distribution(ABC):
    def __init__(self, output_size, name, name_scope):
        self.log = logging.getLogger(self.__class__.__name__ + ":" + str(name_scope))
        self.output_size = output_size
        self.name = name
        self.name_scope = name_scope
        self.batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
        self.latent_vector = tf.random_normal(shape=(self.batch_size, self.output_size))
        self.data_vector_in = tf.placeholder(shape=(None, self.output_size), dtype=tf.float32)

    def sample(self, batch_size):
        return compile_function(inputs=[self.batch_size], outputs=self.data_vector_out)(batch_size)

    def pdf_op(self):
        return self._pdf

    def log_pdf_op(self):
        return self._log_pdf

    def entropy_op(self):
        return self._entropy

    def pdf(self, x):
        return compile_function(inputs=[self.data_vector_in], outputs=self.pdf_op())(x)

    def log_pdf(self, x):
        return compile_function(inputs=[self.data_vector_in], outputs=self.log_pdf_op())(x)

    @property
    def entropy(self):
        return compile_function(inputs=[], outputs=self.entropy_op())()

    @property
    def params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    @property
    def trainable_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_scope)

    def get_params(self):
        return compile_function(inputs=[], outputs=self.params)()

    def build_copy_op(self):
        self._copy_weights = [tf.placeholder(dtype=tf.float32,
                                           shape=p.shape.as_list()) for p in self.params]
        self._copy_op = [tf.assign(p, w) for p, w in zip(self.params, self._copy_weights)]

    def copy_from_distribution(self, target_distribution):
        copy_weights = compile_function(inputs=[], outputs=target_distribution.params)()
        session = tf.get_default_session()
        feed_dict = {cpy: cp for (cpy, cp) in zip(self._copy_weights, copy_weights)}
        session.run(self._copy_op, feed_dict=feed_dict)

    def copy_from_params(self, params):
        session = tf.get_default_session()
        feed_dict = {cpy: cp for (cpy, cp) in zip(self._copy_weights, params)}
        session.run(self._copy_op, feed_dict=feed_dict)

    def diagnostic(self):
        diagnostic = dict()
        diagnostic["entropy"] = float(self.entropy)
        return diagnostic

    @abstractmethod
    def build_copy(self, name_scope):
        pass

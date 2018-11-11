
from utils.tensorflow_utils import compile_function
import numpy as np
import tensorflow as tf


class ObjectiveFunction(object):
    """ Basic class generating differentiable objective functions """

    def __init__(self, dim, name=''):
        """ Init function for the objective function
        Args:
            dim : int, dimensionality of the landscape
        """
        self.dim = dim
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dim))
        self.translation = np.random.uniform(-3, 3, size=(dim,))  # random translation of the landscape
        self.z = self.x - self.translation
        self.out = self.build()
        self.grad = tf.gradients(self.out, self.x)
        self.name = name

    def __str__(self):
        return(self.name + '_' + str(self.dim))

    def build(self):
        """" Builds computational graph. For the generic function, returns a single constant node"""
        raise NotImplementedError('The function build_graph is not implement for base class.')

    def df(self, x):
        """ Plays the role of a black-box first order oracle - returns (f(x),Df(x))
        Args:
            session : tensorflow session
            x : numpy array of query points
        Returns
            f : numpy array of function evals at x
            df : numpy array of gradient evals at x
        """
        f = compile_function(inputs=[self.x], outputs=self.out)(x)
        df = compile_function(inputs=[self.x], outputs=self.grad)(x)
        return f, df[0]

    def f(self, x):
        """ Plays the role of a black-box zeroth order oracle - returns (f(x))
        Args:
            session : tensorflow session
            x : numpy array of query points
        Returns
            f : numpy array of function evals at x
        """
        return compile_function(inputs=[self.x], outputs=self.out)(x)


class Sphere(ObjectiveFunction):
    """ Sphere function class """

    def __init__(self, dim):
        super().__init__(dim, 'sphere')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """" Builds computational graph for f(x) = ||x||^2
        Returns:
            y = ||x||^2
        """
        return tf.reduce_sum(tf.square(self.z), axis=1)


class Rosenbrock(ObjectiveFunction):
    """ Rosenbrock function class """

    def __init__(self, dim):
        super().__init__(dim, 'rosenbrock')
        self.min = np.ones((self.dim,))+self.translation

    def build(self):
        """" Builds computational graph for Rosenbrock's function
        Returns:
            y = SUM_i{ 100*(x_i^2-x_{i+1})^2 + (x_i-1)^2}
        """
        self.sum_list = []
        for i in range(self.dim - 1):
            zi = 100 * tf.square(tf.square(self.z[:, i]) -
                                 self.z[:, i + 1]) + tf.square(self.z[:, i] - 1)
            self.sum_list.append(zi)
        return tf.add_n(self.sum_list)


class SumOfDifPower(ObjectiveFunction):
    """ Sum of different power function """

    def __init__(self, dim):
        super().__init__(dim, 'sumdifpower')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """" Builds computational graph for sum of difference power functions

        Returns:
            y = SUM_i{ |z_i|^(2+4*(i-1)/(d-1) }
        """
        self.sum_list = []
        for i in range(self.dim):
            exp = 2 + 10 * i / (self.dim - 1)
            zi = tf.abs(self.z[:, i])
            self.sum_list.append(zi**exp)

        return tf.sqrt(tf.add_n(self.sum_list))


class Rastrigin(ObjectiveFunction):
    """  Rastrigin's function class """

    def __init__(self, dim):
        super().__init__(dim, 'rastrigin')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """" Builds computational graph for Rastrigin's function

        Returns:
            y = SUM_i{ |z_i|^(2+4*(i-1)/(d-1) }
        """
        self.sum_list = []
        for i in range(self.dim):
            zi = tf.cos(2 * np.pi * self.z[:, i])
            self.sum_list.append(zi)
        return 10 * (self.dim - tf.add_n(self.sum_list)) + \
            tf.reduce_sum(tf.square(self.z), axis=1)


class Discus(ObjectiveFunction):
    """ Discus function class """

    def __init__(self, dim):
        super().__init__(dim, 'discus')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """ Builds computational graph for Discus function

        Returns:
            y = x_0^2 + SUM_i{10^-6 x_i ^2}
        """
        self.sum_list = []
        for i in range(self.dim):
            lambd = 1 if i == 0 else 1e-6
            zi = lambd * tf.square(self.z[:, i])
            self.sum_list.append(zi)
        return tf.add_n(self.sum_list)


class Cigar(ObjectiveFunction):
    """ Cigar function class """

    def __init__(self, dim):
        super().__init__(dim, 'cigar')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """ Builds computational graph for Cigar function

        Returns:
            y = 1e-6 x_0^2 + SUM_i{ x_i ^2}
        """
        self.sum_list = []
        for i in range(self.dim):
            lambd = 1e-6 if i == 0 else 1
            zi = lambd * tf.square(self.z[:, i])
            self.sum_list.append(zi)
        return tf.add_n(self.sum_list)


class Ellipsoid(ObjectiveFunction):
    """ Ellispoid function class """

    def __init__(self, dim):
        super().__init__(dim, 'ellipsoid')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """ Builds computational graph for Ellipsoid function """
        self.sum_list = []
        for i in range(self.dim):
            lambd = 10**(-6 * i / (self.dim - 1))
            zi = lambd * tf.square(self.z[:, i])
            self.sum_list.append(zi)
        return tf.add_n(self.sum_list)


class SixHumpCamel(ObjectiveFunction):
    """ Six Hump Camel function class """

    def __init__(self, dim):
        super().__init__(dim, 'sixhumpcamel')
        self.min = np.zeros((self.dim,))+self.translation
        self.min[0] = self.min[0] + 0.0898
        self.min[1] = self.min[1] - 0.7126

    def build(self):
        """ Builds computational graph for SHC function
        Returns:
            y = (4-2.1x[0]**2+x[0]**4/3) + x[0]x[2] + (-4+4*x[1]**2)x[1]**2
        """
        self.sum_list = []
        z1 = (4 - 2.1 * tf.square(self.z[:, 0]) + tf.square(tf.square(self.z[:, 0])) / 3) * tf.square(
            self.z[:, 0]) + self.z[:, 0] * self.z[:, 1] + (-4 + 4 * tf.square(self.z[:, 1])) * tf.square(self.z[:, 1])
        self.sum_list.append(z1)
        for i in np.arange(2, self.dim):
            zi = tf.square(self.z[:, i])
            self.sum_list.append(zi)
        return tf.add_n(self.sum_list) + 1.031628453489877


class Ackley(ObjectiveFunction):
    """ Ackley's function class """

    def __init__(self, dim):
        super().__init__(dim, 'ackley')
        self.min = np.zeros((self.dim,))+self.translation

    def build(self):
        """ Builds computational graph for Ackley's function """
        a = 20
        b = 0.2
        c = 2 * np.pi

        z1 = -a * tf.exp(-b * tf.sqrt((1.0 / self.dim) *
                                      tf.reduce_sum(tf.square(self.z), axis=1)))
        z2 = -tf.exp((1.0 / self.dim) * tf.reduce_sum(tf.cos(c * self.z), axis=1))
        z3 = a + np.exp(1)

        return z1 + z2 + z3


class Styblinski(ObjectiveFunction):
    """ Styblinski function class """

    def __init__(self, dim):
        super().__init__(dim, 'styblinski')
        self.min = -2.903534*np.ones((self.dim,))+self.translation

    def build(self):
        """ Builds computational graph for Styblinski's function """
        self.sum_list = []
        for i in range(self.dim):
            zi = tf.square(tf.square(self.z[:, i])) - 16 * \
                tf.square(self.z[:, i]) + 5 * self.z[:, i]
            self.sum_list.append(zi)
        return 0.5 * tf.add_n(self.sum_list) + 39.16616570377142 * self.dim
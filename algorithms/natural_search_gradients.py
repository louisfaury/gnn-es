
import tensorflow as tf
import numpy as np
from algorithms.base_algo import BaseAlgo
from utils.tensorflow_utils import compile_function, flatten_tensor_list, reshape_vector_to_tensor_list
from scipy.sparse.linalg import cg, LinearOperator


class NaturalSearchGradients(BaseAlgo):
    def __init__(self, config, distribution):
        super().__init__(config, distribution)

        # hyper-parameters
        self.lr = self.config["lr"]
        self.fischer_sample_size = config["fischer_sample_size"]
        self.fischer_damping = config["fischer_damping"]

        # create surrogate loss and related op
        cost = tf.placeholder(dtype=tf.float32, shape=(None,), name='cost')
        data_vector = distribution.data_vector_in
        surrogate_loss = tf.reduce_mean(cost * distribution.log_pdf_op())
        self.loss = compile_function(inputs=[cost, data_vector], outputs=surrogate_loss)

        distribution_params = distribution.trainable_params
        self.distribution_params_shape_list = [tensor.get_shape() for tensor in distribution_params]

        surrogate_loss_gradients = tf.gradients(surrogate_loss, distribution_params)
        flat_surrogate_loss_gradients = flatten_tensor_list(surrogate_loss_gradients)
        self.distribution_params_dim = flat_surrogate_loss_gradients.get_shape()[0]
        self.flat_surrogate_loss_gradients = compile_function(inputs=[cost, data_vector], outputs=flat_surrogate_loss_gradients)

        # hessian-vector product
        vector_placeholder = [tf.placeholder(dtype=tf.float32, shape=tensor_shape) for tensor_shape in self.distribution_params_shape_list]
        negative_log_pdf = tf.reduce_mean(-self.distribution.log_pdf_op())
        negative_log_pdf_gradients = tf.gradients(negative_log_pdf, distribution_params)
        negative_log_pdf_gradients_vector_product = [tf.reduce_sum(g*v) for g,v in zip(negative_log_pdf_gradients, vector_placeholder)]

        fischer_vector_product = tf.gradients(negative_log_pdf_gradients_vector_product, distribution_params, stop_gradients=vector_placeholder)
        flat_fischer_vector_product = flatten_tensor_list(fischer_vector_product)
        self.flat_fischer_vector_product = compile_function(inputs=[data_vector] + vector_placeholder, outputs=flat_fischer_vector_product)

        # update operations
        delta_params_placeholder = [tf.placeholder(dtype=tf.float32, shape=tensor_shape) for tensor_shape in self.distribution_params_shape_list]
        update_params_op = tf.group([tf.assign(p, p + d) for (p, d) in zip(distribution_params, delta_params_placeholder)])
        self.update_params = compile_function(inputs=delta_params_placeholder, outputs=update_params_op)

    def fit(self, samples):
        diagnostic = dict()

        fit_data = np.reshape(samples["data"], (-1, self.output_size))
        fit_cost = samples["cost"]
        avg_cost = np.mean(fit_cost)
        min_cost = np.min(fit_cost)

        normalize_fit_cost = (fit_cost - np.mean(fit_cost)) / (1e-5 + np.std(fit_cost))
        loss_before = self.loss(normalize_fit_cost, fit_data)

        fischer_sampled_data = self.distribution.sample(self.fischer_sample_size)
        fischer_vector_product_on_data = lambda vec:  self.damped_hessian_vector_product(fischer_sampled_data, vec)
        fischer_vector_product_linear_operator = LinearOperator(shape=(self.distribution_params_dim, self.distribution_params_dim),
                                                                matvec=fischer_vector_product_on_data)
        flat_surrogate_loss_gradients = self.flat_surrogate_loss_gradients(normalize_fit_cost, fit_data)
        delta_params, _ = cg(fischer_vector_product_linear_operator, -flat_surrogate_loss_gradients, maxiter=50)
        delta_params *= self.lr
        tensor_delta_params = reshape_vector_to_tensor_list(delta_params, self.distribution_params_shape_list)
        hessian_delta_product = self.hessian_vector_product(fischer_sampled_data, delta_params)
        self.update_params(*tensor_delta_params)

        loss_after = self.loss(normalize_fit_cost, fit_data)

        # update the damping parameter
        # q_theta =  loss_before + np.sum(delta_params * flat_surrogate_loss_gradients) + 0.5 * np.sum(
        #     delta_params * hessian_delta_product)
        # reduction_ratio = (loss_after-loss_before)/(q_theta-loss_before)

        # if reduction_ratio < 0.25:
        #     self.fischer_damping *= 1.5
        # if reduction_ratio > 0.75:
        #     self.fischer_damping *= 2. / 3.

        diagnostic["loss_before"] = float(loss_before)
        diagnostic["loss_after"] = float(loss_after)
        diagnostic["mean_cost"] = float(avg_cost)
        diagnostic["min_cost"] = float(min_cost)
        diagnostic["fischer_damping"] = self.fischer_damping
        diagnostic.update(self.distribution.diagnostic())

        return diagnostic

    def damped_hessian_vector_product(self, data, vector):
        hessian_vec_product = self.hessian_vector_product(data ,vector)
        damped_hessian_vec_product = hessian_vec_product + self.fischer_damping*vector
        return damped_hessian_vec_product

    def hessian_vector_product(self, data, vector):
        assert (np.shape(vector) == (self.distribution_params_dim,))
        tensorized_vector = reshape_vector_to_tensor_list(vector, self.distribution_params_shape_list)
        hessian_vec_product = self.flat_fischer_vector_product(data, *tensorized_vector)
        return hessian_vec_product

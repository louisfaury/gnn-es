
from algorithms.base_algo import BaseAlgo
from utils.tensorflow_utils import compile_function
import tensorflow as tf
import numpy as np
from optimizers.optimizer import create_optimizer


class VanillaSearchGradients(BaseAlgo):
    def __init__(self, config, distribution):
        super().__init__(config, distribution)

        # create optimizer
        optimizer_config = config["optimizer"]
        optimizer_name = optimizer_config["name"]
        optimizer_lr = optimizer_config["lr"]
        self.optimizer = create_optimizer(optimizer_name, optimizer_lr)

        # create surrogate loss
        cost = tf.placeholder(dtype=tf.float32, shape=(None,), name='cost')
        data_vector = distribution.data_vector_in
        surrogate_loss = tf.reduce_mean(cost*distribution.log_pdf_op())
        self.loss = compile_function(inputs=[cost, data_vector], outputs=surrogate_loss)

        # create optimize op
        optimize = self.optimizer.minimize(surrogate_loss)
        self.optimize = compile_function(inputs=[cost, data_vector], outputs=optimize)

    def fit(self, samples):
        diagnostic = dict()

        fit_data = np.reshape(samples["data"], (-1, self.output_size))
        fit_cost = samples["cost"]
        avg_cost = np.mean(fit_cost)
        min_cost = np.min(fit_cost)

        normalize_fit_cost = (fit_cost - np.mean(fit_cost)) / (1e-5 + np.std(fit_cost))
        loss_before = self.loss(normalize_fit_cost, fit_data)
        self.optimize(normalize_fit_cost, fit_data)
        loss_after = self.loss(normalize_fit_cost, fit_data)

        diagnostic["loss_before"] = float(loss_before)
        diagnostic["loss_after"] = float(loss_after)
        diagnostic["mean_cost"] = float(avg_cost)
        diagnostic["min_cost"] = float(min_cost)
        diagnostic.update(self.distribution.diagnostic())

        return diagnostic


from algorithms.base_algo import BaseAlgo
from utils.tensorflow_utils import compile_function
from optimizers.optimizer import create_optimizer
import tensorflow as tf
import numpy as np


class AdaptiveVanillaSearchGradients(BaseAlgo):
    def __init__(self, config, distribution):
        super().__init__(config, distribution)

        # create step-size variable and optimizer
        self.lr = tf.Variable(float(config["optimizer"]["lr"]), dtype=tf.float32, trainable=False)
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=())
        self.assign_lr = compile_function(inputs=[self.lr_placeholder], outputs=tf.assign(self.lr, self.lr_placeholder))
        self.get_lr = compile_function(inputs=[], outputs=self.lr)
        self.optimizer = create_optimizer(config["optimizer"]["name"], self.lr)

        # surrogate loss
        cost = tf.placeholder(dtype=tf.float32, shape=(None,), name='cost')
        data_vector = distribution.data_vector_in
        surrogate_loss = tf.reduce_mean(cost*distribution.log_pdf_op())
        self.loss = compile_function(inputs=[cost, data_vector], outputs=surrogate_loss)

        # optimization op
        optimize = self.optimizer.minimize(surrogate_loss)
        self.optimize = compile_function(inputs=[cost, data_vector], outputs=optimize)

        # kl divergence op
        self.distribution_copy = self.distribution.build_copy()
        kl_op = self.distribution.build_kl(self.distribution_copy)
        self.kl = compile_function(inputs=[], outputs=kl_op)
        self.kl_step_size = config["kl_step_size"]

    def fit(self, samples):
        diagnostic = dict()

        self.distribution_copy.copy_from_distribution(self.distribution)

        fit_data = np.reshape(samples["data"], (-1, self.output_size))
        fit_cost = samples["cost"]
        avg_cost = np.mean(fit_cost)
        min_cost = np.min(fit_cost)

        normalize_fit_cost = (fit_cost - np.mean(fit_cost)) / (1e-5 + np.std(fit_cost))
        loss_before = self.loss(normalize_fit_cost, fit_data)
        self.optimize(normalize_fit_cost, fit_data)
        loss_after = self.loss(normalize_fit_cost, fit_data)
        kl_old_new = self.kl()
        self.update_lr(kl_old_new)

        diagnostic["loss_before"] = float(loss_before)
        diagnostic["loss_after"] = float(loss_after)
        diagnostic["mean_cost"] = float(avg_cost)
        diagnostic["min_cost"] = float(min_cost)
        diagnostic["kl"] = float(kl_old_new)
        diagnostic["step_size"] = float(self.get_lr())
        diagnostic.update(self.distribution.diagnostic())

        return diagnostic

    def update_lr(self, kl):
        lr = self.get_lr()
        if kl > 2*self.kl_step_size:
            lr /= 1.5
        elif kl < self.kl_step_size/2:
            lr *= 1.5
        lr = np.clip(lr, 0.001*self.config["optimizer"]["lr"], 1000*self.config["optimizer"]["lr"])
        self.assign_lr(lr)

from algorithms.base_algo import BaseAlgo
import logging
import numpy as np
import tensorflow as tf
from utils.tensorflow_utils import compile_function, compile_scipy_optimizer
from utils.utils import get_or_default, log


class TrustRegionSearchGradients(BaseAlgo):
    def __init__(self, config, distribution):
        super().__init__(config, distribution)

        # define loss
        cost = tf.placeholder(dtype=tf.float32, shape=(None,), name='cost')
        data_vector = distribution.data_vector_in
        old_prob = tf.placeholder(dtype=tf.float32, shape=(None,), name='old_log_pdf')
        importance_ratio = tf.div(self.distribution.pdf_op(), old_prob+1e-5)
        surrogate_loss = tf.reduce_mean(cost*importance_ratio)
        self.loss = compile_function(inputs=[cost, old_prob, data_vector], outputs=surrogate_loss)

        # kl divergence op
        self.sampling_distribution = self.distribution.build_copy()
        kl_op = self.distribution.build_kl(self.sampling_distribution)
        self.kl = compile_function(inputs=[], outputs=kl_op)

        # get (default) hyper-parameters
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt_max_iter = get_or_default(config=config,
                                           key="optimization_max_iter",
                                           default=50,
                                           logger=self.logger)
        self.pen_max_iter = get_or_default(config=config,
                                           key="penalty_max_iter",
                                           default=20,
                                           logger=self.logger)
        self.min_penalty = get_or_default(config=config,
                                          key="min_penalty",
                                          default=1e-2,
                                          logger=self.logger)
        self.max_penalty = get_or_default(config=config,
                                          key="max_penalty",
                                          default=1e5,
                                          logger=self.logger)
        self.init_penalty = get_or_default(config=config,
                                           key="init_penalty",
                                           default=1.,
                                           logger=self.logger)
        self.max_kl = get_or_default(config=config,
                                     key="kl_step_size",
                                     default=0.01,
                                     logger=self.logger)
        self.update_penalty_coeff = get_or_default(config=config,
                                                   key="update_penalty_coeff",
                                                   default=2.0,
                                                   logger=self.logger)

        # define the penalized objective operation
        penalty_var = tf.Variable(self.init_penalty, trainable=False, dtype=tf.float32)
        penalty_placeholder = tf.placeholder(shape=(), dtype=tf.float32)
        assign_penalty_op = tf.assign(penalty_var, penalty_placeholder)
        self.assign_penalty = compile_function(inputs=[penalty_placeholder], outputs=assign_penalty_op)
        self.penalty = compile_function(inputs=[], outputs=penalty_var)
        penalized_loss_op = surrogate_loss + penalty_var * kl_op

        # scipy interfance minimization
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss=penalized_loss_op,
                                                           var_list=self.distribution.trainable_params,
                                                           options={'maxiter': self.opt_max_iter})
        self.minimize = compile_scipy_optimizer(optimizer=optimizer,
                                                inputs=[cost, old_prob, data_vector])

    def fit(self, samples):
        diagnostic = dict()

        self.sampling_distribution.copy_from_distribution(self.distribution)

        fit_data = np.reshape(samples["data"], (-1, self.output_size))
        fit_cost = samples["cost"]
        avg_cost = np.mean(fit_cost)
        min_cost = np.min(fit_cost)

        probs = self.sampling_distribution.pdf(fit_data)

        normalize_fit_cost = (fit_cost - np.mean(fit_cost)) / (1e-5 + np.std(fit_cost))
        loss_before = self.loss(normalize_fit_cost, probs, fit_data)
        self.optimize_trust_region(normalize_fit_cost, probs, fit_data)
        loss_after = self.loss(normalize_fit_cost, probs, fit_data)
        kl_old_new = self.kl()
        kl_penalty = self.penalty()

        diagnostic["loss_before"] = float(loss_before)
        diagnostic["loss_after"] = float(loss_after)
        diagnostic["mean_cost"] = float(avg_cost)
        diagnostic["min_cost"] = float(min_cost)
        diagnostic["kl"] = float(kl_old_new)
        diagnostic["kl_penalty"] = float(kl_penalty)
        diagnostic.update(self.distribution.diagnostic())

        return diagnostic

    def optimize_trust_region(self, score, probs, data):
        penalty = self.penalty()
        target_params = None
        update_penalty_factor = None

        for pen_iter in range(self.pen_max_iter):
            stop = False
            self.distribution.copy_from_distribution(self.sampling_distribution)
            self.assign_penalty(penalty)
            try:
                self.minimize(score, probs, data)
                kl = self.kl()
            except:
                kl = self.max_kl+1

            print(penalty, kl)
            if kl > self.max_kl or np.isnan(kl):
                if update_penalty_factor is None:
                    update_penalty_factor = self.update_penalty_coeff
                if update_penalty_factor < 1:
                    stop = True

            if kl < self.max_kl:
                if update_penalty_factor is None:
                    update_penalty_factor = 1./self.update_penalty_coeff
                if update_penalty_factor > 1:
                    stop = True
                target_params = self.distribution.get_params()

            penalty *= update_penalty_factor
            if penalty < self.min_penalty:
                stop = True
            if penalty > self.max_penalty:
                pen_iter = self.pen_max_iter - 1
                stop = True

            if stop or pen_iter == self.pen_max_iter-1:
                if pen_iter == self.pen_max_iter-1:
                    log("!! Could not find appropriate penalty !! Copying sampling distribution !!", self.logger)
                    target_params = self.sampling_distribution.get_params()
                    self.assign_penalty(self.init_penalty)
                self.distribution.copy_from_params(target_params)
                break

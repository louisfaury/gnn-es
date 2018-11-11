
from distributions.distribution_factory import create_distribution
from objectives.objective_factory import create_objective
from samplers.sampler_factory import create_sampler
from algorithms.algo_factory import create_algorithm
import os
import argparse
import pprint as pp
import json
import tensorflow as tf
import numpy as np


def run_experiments(config):
    seeds = config["seeds"]
    dimension = config["dimensions"]
    max_eval_per_run = config["max_eval_per_run"]

    outputs_dict = dict()
    outputs_config_dict = dict()
    outputs_config_dict["objective"] = config["objective"]
    outputs_config_dict["distribution"] = config["distribution"]
    outputs_config_dict["algo"] = config["algo"]
    outputs_config_dict["sampler"] = config["sampler"]
    outputs_dict["config"] = outputs_config_dict

    output_results_dict = dict()
    for seed in seeds:
        results = run_one_experiment(config, dimension, seed, max_eval_per_run)
        output_results_dict[seed] = results
    outputs_dict["results"] = output_results_dict

    return outputs_dict


def run_one_experiment(config, output_size, seed, max_evals):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    session = tf.Session()

    with session.as_default():
        distribution_config = config["distribution"]
        distribution = create_distribution(distribution_config, output_size)

        sampler_config = config["sampler"]
        sampler = create_sampler(sampler_config, distribution)
    
        algo_config = config["algo"]
        algo = create_algorithm(algo_config, distribution)

        objective_config = config["objective"]
        objective = create_objective(objective_config, output_size)

        session.run(tf.global_variables_initializer())
        results = dict()
        evals = 0
        iters = 0
        while evals<max_evals:
            samples = dict()
            queries, new_evals = sampler.sample()
            evals += new_evals
            scores = objective.f(queries)
            samples["data"] = queries
            samples["cost"] = scores

            diagnostic = algo.fit(samples)
            diagnostic["evals"] = evals
            print_diagnostics(iters, diagnostic)
            iters += 1
            results[evals] = diagnostic
    return results


def print_diagnostics(iteration, diagnostics):
    print("Iteration %i" % iteration)
    print("=================")
    for key in diagnostics.keys():
        print(key,":", diagnostics[key])
    print("=================")
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ES with Invertible Networks')
    parser.add_argument('--config_name',
                        help='Name for the configuration file',
                        type=str)
    parser.add_argument('--output_name',
                        help='Name for the result file',
                        type=str)
    args = vars(parser.parse_args())
    pp.pprint(args)

    config_file_dir = 'config'
    if not os.path.isdir(config_file_dir):
        os.makedirs(config_file_dir)
    output_file_dir = 'logs'
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)

    config_file_name = args["config_name"]
    config_file = os.path.join(config_file_dir, config_file_name)
    with open(config_file, 'r') as f:
        config = json.load(f)
    results = run_experiments(config)

    output_file_name = args["output_name"]
    output_file = os.path.join(output_file_dir, output_file_name)
    with open(output_file, 'w') as f:
        json.dump(results, f)



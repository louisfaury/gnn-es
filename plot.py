
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def mean_curve(flist, xmin, xmax):
    x = np.arange(xmin+1, xmax-1, (xmax-xmin)/1000)
    return x, savgol_filter(np.mean([f(x) for f in flist], axis=0), window_length=31, polyorder=1)


parser = argparse.ArgumentParser(description='ES with Invertible Networks')
args = vars(parser.parse_args())

data_file_dir = 'logs'
if not os.path.isdir(data_file_dir):
    raise ValueError('Unknown data directory')

img_dir = 'img'
if not os.path.isdir(img_dir):
    os.makedirs(img_dir)

# envs
objective_names = ['sphere', 'rastrigin', 'ackley', 'rosenbrock']

# plot config
plt.figure(figsize=(8, 6))
colors = {'nvp': 'C0', 'gaussian': 'C1', 'nice': 'C2'}
markers = {'vsg':'>', 'a-vsg':'s', 'trsg':'v', 'nsg':'x'}

for i, obj in enumerate(objective_names):
    plt.figure(figsize=(12, 8))
    ax_min = plt.subplot(2, 2, 1)
    ax_min.set_yscale('log', nonposy='clip')
    ax_min.margins(0.01)
    plt.title('min cost')
    ax_mean = plt.subplot(2, 2, 3)
    ax_mean.set_yscale('log', nonposy='clip')
    ax_mean.margins(0.01)
    plt.title('mean cost')
    ax_entropy = plt.subplot(2, 2, 2)
    ax_entropy.margins(0.01)
    plt.title('entropy')
    ax_loss = plt.subplot(2, 2, 4)
    ax_loss.margins(0.01)
    plt.title('surrogate loss')
    plt.tight_layout(pad=2)

    for config_name in os.listdir(data_file_dir):
        if os.path.isdir(os.path.join(data_file_dir, config_name)):
            continue
        with open(os.path.join(data_file_dir,config_name)) as file:
            data = json.load(file)
            config = data["config"]
            obj_name = config["objective"]["name"]
            if not obj_name == obj:
                pass
            else:
                algo_name = config["algo"]["name"]
                distribution_name = config["distribution"]["name"]
                if distribution_name == 'ddm':
                    if config["distribution"]["invertible_network"]["activation"] == 'identity':
                        distribution_name = 'deep_gaussian'
                    else:
                        distribution_name = 'deep_density'
                results = data["results"]
                min_interpolated_curves = []
                mean_interpolated_curves = []
                entropy_interpolated_curves = []
                loss_interpolated_curves = []

                min_evals = -np.inf
                max_evals = np.inf

                for seed in results.keys():
                    number_calls = np.array([])
                    min_values = np.array([])
                    mean_min_values = np.array([])
                    entropy_values = np.array([])
                    loss_values = np.array([])
                    result_per_seed = results[seed]
                    for eval_key in result_per_seed:
                        number_calls = np.concatenate((number_calls, [int(eval_key)]), axis=0)
                        min_values = np.concatenate((min_values, [result_per_seed[eval_key]["min_cost"]]))
                        mean_min_values = np.concatenate((mean_min_values, [result_per_seed[eval_key]["mean_cost"]]))
                        entropy_values = np.concatenate((entropy_values, [result_per_seed[eval_key]["entropy"]]))
                        loss_values = np.concatenate((loss_values, [result_per_seed[eval_key]["loss_after"]]))

                    min_evals = np.max([min_evals, np.min(number_calls)])
                    max_evals = np.min([max_evals, np.max(number_calls)])
                    min_interpolated_curves.append(interp1d(number_calls, min_values))
                    mean_interpolated_curves.append(interp1d(number_calls, mean_min_values))
                    entropy_interpolated_curves.append(interp1d(number_calls, entropy_values))
                    loss_interpolated_curves.append(interp1d(number_calls, loss_values))
                    sort_index = np.argsort(number_calls)

                    # ax_min.plot(number_calls[sort_index], min_values[sort_index],
                    #             color=colors[algo_name.lower()], marker=markers[distribution_name.lower()],
                    #             alpha=0.05, markevery=20)
                    # ax_mean.plot(number_calls[sort_index], mean_min_values[sort_index],
                    #              color=colors[algo_name.lower()], marker=markers[distribution_name.lower()],
                    #              alpha=0.05, markevery=20)
                    # ax_entropy.plot(number_calls[sort_index], entropy_values[sort_index],
                    #                 color=colors[algo_name.lower()], marker=markers[distribution_name.lower()],
                    #                 alpha=0.05, markevery=20)
                    # ax_loss.plot(number_calls[sort_index], loss_values[sort_index],
                    #              color=colors[algo_name.lower()], marker=markers[distribution_name.lower()],
                    #              alpha=0.05, markevery=20)

                # compute and plot mean curves
                mean_min_evals, mean_min_values = mean_curve(min_interpolated_curves, min_evals, max_evals)
                ax_min.plot(mean_min_evals, mean_min_values, color=colors[distribution_name.lower()], linewidth=1,
                            marker=markers[algo_name.lower()], label=algo_name.lower() +'_' + distribution_name.lower(),
                            markevery=50)

                mean_mean_evals, mean_mean_values = mean_curve(mean_interpolated_curves, min_evals, max_evals)
                ax_mean.plot(mean_mean_evals, mean_mean_values, color=colors[distribution_name.lower()],linewidth=1,
                             marker=markers[algo_name.lower()], label=algo_name.lower() +'_' + distribution_name.lower(),
                             markevery=50)

                mean_entropy_evals, mean_entropy_values = mean_curve(entropy_interpolated_curves, min_evals, max_evals)
                ax_entropy.plot(mean_entropy_evals, mean_entropy_values, color=colors[distribution_name.lower()],
                                marker=markers[algo_name.lower()], linewidth=1, label=algo_name.lower() +'_' + distribution_name.lower(),
                                markevery=50)

                mean_loss_evals, mean_loss_values = mean_curve(loss_interpolated_curves, min_evals, max_evals)
                ax_loss.plot(mean_loss_evals, mean_loss_values, color=colors[distribution_name.lower()],
                             marker=markers[algo_name.lower()], linewidth=1, label=algo_name.lower() +'_' + distribution_name.lower(),
                             markevery=50)

    ax_min.set_xlabel('# evaluations')
    ax_min.set_ylabel('objective value')
    ax_min.legend(loc=1)

    plt.savefig(os.path.join(img_dir, algo_name.lower() + '_' + obj.lower()), bbox_inches='tight')
    plt.gcf().clear()
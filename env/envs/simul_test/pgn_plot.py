import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from collections import defaultdict
from matplotlib.ticker import ScalarFormatter, MaxNLocator


class ScalarFormatterForceFormat_zero(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.0f"

class ScalarFormatterForceFormat_one(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

class ScalarFormatterForceFormat_two(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Data type to visualize", type=int, required=True)
parser.add_argument("-c", "--category", help="Category to visualize", type=str, default="all")
parser.add_argument("-r", "--record", help="Print result or not", type=bool, default=False)
parser.add_argument("-p", "--plot", help="Plot result or not", type=bool, default=False)
args = parser.parse_args()
data_type = args.type
category = args.category
record_result = args.record
plot_result = args.plot

assert category in ["all", "sr", "time", "distance"], "Unavailable category to plot"

evaluate_seeds = [37, 143, 534]
grid_sizes = [2.5, 3.0, 4.0, 5.0]
evaluation_metrics = ["Success rate", "Time", "Distance"]
# result_folder_path = "/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/Simple_point_goal_nav/Result"
result_folder_path = "/Users/kimyunho/Desktop/Simple_point_goal_nav/Result"

if data_type == 1:
    # Main: PGN result in open field with scattered obstacles
    # data_names = ["CVAE_729_100", "CVAE_125_100", "Naive_1000", "CWM_64", "Oracle"]
    # label_names = {"CVAE_729_100": "Hybrid", "CVAE_125_100": "Hybrid_light", "Naive_1000": "Naive", "CWM_64": "CWM", "Oracle": "Oracle"}
    # colors = {"CVAE_729_100": "red", "CVAE_125_100": "blue", "Naive_1000": "green", "CWM_64": "orange", "Oracle": "violet"}
    data_names = ["CVAE_729_100", "CVAE_125_100", "Naive_1000", "CWM_64"]
    label_names = {"CVAE_729_100": "Hybrid", "CVAE_125_100": "Hybrid_light", "Naive_1000": "Naive", "CWM_64": "CWM"}
    colors = {"CVAE_729_100": "red", "CVAE_125_100": "blue", "Naive_1000": "green", "CWM_64": "orange"}
elif data_type == 2:
    # Ablation1: importance of predicted P_col in planning
    data_names = ["Naive_64", "CWM_64"]
    label_names = {"Naive_64": "Naive_64", "CWM_64": "CWM"}
    colors = {"Naive_64": "red", "CWM_64": "orange"}
    # Naive_64 (o) / CWM (x)
elif data_type == 3:
    # Ablation2: importance of number of sampled trajectories for sampling based path optimizer
    data_names = ["Naive_64", "Naive_216", "Naive_512", "Naive_1000"]
    colors = {"Naive_64": "orange", "Naive_216": "green", "Naive_512": "blue", "Naive_1000": "red"}
    # Naive_64 (o) / Naive_216 (o) / Naive_512 (o) / Naive_1000 (x)
elif data_type == 4:
    # Ablation3: effect of learned sampling distribution in planning
    data_names = ["CVAE_729_100", "Naive_729", "CVAE_0_100"]
    label_names = {"Naive_729": "only Naive", "CVAE_0_100": "only CVAE", "CVAE_729_100": "Hybrid"}
    colors = {"Naive_729": "green", "CVAE_0_100": "orange", "CVAE_729_100": "red"}
    # Naive_729 (o) / CVAE_0_100 (o) / CVAE_729_100 (x)


# https://matplotlib.org/2.0.2/examples/color/named_colors.html :  matplotlib color codes

#  Set empty container to save result
total_result_dict = defaultdict(dict)
for data_name in data_names:
    for evaluation_metric in evaluation_metrics:
        total_result_dict[data_name][evaluation_metric + "_mean"] = np.zeros(len(grid_sizes))
        total_result_dict[data_name][evaluation_metric + "_std"] = np.zeros(len(grid_sizes))

# Generate empty text file to record data
if record_result:
    open(f"pgn_result_{data_type}.txt", "w").close()

# Load recorded result & Compute mean and std
for data_name in data_names:
    if record_result:
        with open(f"pgn_result_{data_type}.txt", "a") as f:
            f.write("===========================================\n")
            f.write(data_name + "\n\n")
    for i, grid_size in enumerate(grid_sizes):
        list_sr = []
        list_time = []
        list_distance = []

        for evaluate_seed in evaluate_seeds:
            specific_folder_name = f"{data_name}_{str(evaluate_seed)}"
            file_name = f"{str(grid_size)}_grid_result.json"
            final_path = f"{result_folder_path}/{specific_folder_name}/{file_name}"
            with open(final_path, "r") as f:
                recorded_result = json.load(f)

            list_sr.append(recorded_result["SR"]["ratio"])
            list_time.append(recorded_result["Time"]["mean"])
            list_distance.append(recorded_result["Distance"]["mean"])

        list_sr = np.array(list_sr)
        list_time = np.array(list_time)
        list_distance = np.array(list_distance)

        total_result_dict[data_name][evaluation_metrics[0] + "_mean"][i] = np.mean(list_sr)
        total_result_dict[data_name][evaluation_metrics[0] + "_std"][i] = np.std(list_sr)
        total_result_dict[data_name][evaluation_metrics[1] + "_mean"][i] = np.mean(list_time)
        total_result_dict[data_name][evaluation_metrics[1] + "_std"][i] = np.std(list_time)
        total_result_dict[data_name][evaluation_metrics[2] + "_mean"][i] = np.mean(list_distance)
        total_result_dict[data_name][evaluation_metrics[2] + "_std"][i] = np.std(list_distance)

        # Record result
        if record_result:
            with open(f"pgn_result_{data_type}.txt", "a") as f:
                f.write(f"Grid_{str(grid_size)}:\n")
                f.write(f"SR: {round(np.mean(list_sr), 3)} ({round(np.std(list_sr), 3)}) ")
                f.write(f"Time: {round(np.mean(list_time), 3)} ({round(np.std(list_time), 3)}) ")
                f.write(f"Distance: {round(np.mean(list_distance), 3)} ({round(np.std(list_distance), 3)})\n")

if plot_result:
    # Plot result
    if category == "all":
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
        for i, evaluation_metric in enumerate(evaluation_metrics):
            for data_name in data_names:
                mean_data = total_result_dict[data_name][evaluation_metric + "_mean"]
                std_data = total_result_dict[data_name][evaluation_metric + "_std"]
                if data_type == 1 or data_type == 4:
                    ax[i].plot(range(len(grid_sizes)), mean_data, label=label_names[data_name], color=colors[data_name])
                else:
                    ax[i].plot(range(len(grid_sizes)), mean_data, label=data_name, color=colors[data_name])
                ax[i].fill_between(range(len(grid_sizes)), mean_data + std_data, mean_data - std_data, alpha=0.2, color=colors[data_name])

            ax[i].xaxis.set_ticks([0, 1, 2, 3])
            ax[i].set_xticklabels(grid_sizes)
            ax[i].set_xlabel("Grid size [m]")

            if i == 1:
                ax[i].yaxis.set_major_formatter(ScalarFormatterForceFormat_zero())
            elif i == 2:
                ax[i].yaxis.set_major_formatter(ScalarFormatterForceFormat_one())
            else:
                ax[i].yaxis.set_major_formatter(ScalarFormatterForceFormat_two())
            ax[i].yaxis.set_major_locator(MaxNLocator(5))

            if i == 0:
                ax[i].set_ylabel("Success rate")
            elif i == 1:
                ax[i].set_ylabel("Time [s]")
            else:
                ax[i].set_ylabel("Distance [m]")

            ax[i].set_title(evaluation_metric, fontsize=20)
            # ax[i].legend()

        lines, labels = fig.axes[-1].get_legend_handles_labels()

        fig.legend(lines, labels, loc='lower center', ncol=len(data_names), bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True)
        # plt.show()
        plt.savefig(f"Result_{data_type}.png", bbox_inches='tight')
        plt.clf()
        plt.close()

    else:
        if category == "sr":
            evaluation_metric = "Success rate"
            y_label = "Success rate"
        elif category == "time":
            evaluation_metric = "Time"
            y_label = "Time [s]"
        else:
            evaluation_metric = "Distance"
            y_label = "Distance [m]"

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        for data_name in data_names:
            mean_data = total_result_dict[data_name][evaluation_metric + "_mean"]
            std_data = total_result_dict[data_name][evaluation_metric + "_std"]
            if data_type == 1 or data_type == 2 or data_type == 4:
                ax.plot(range(len(grid_sizes)), mean_data, label=label_names[data_name], color=colors[data_name])
            else:
                ax.plot(range(len(grid_sizes)), mean_data, label=data_name, color=colors[data_name])
            ax.fill_between(range(len(grid_sizes)), mean_data + std_data, mean_data - std_data, alpha=0.2, color=colors[data_name])

        ax.xaxis.set_ticks([0, 1, 2, 3])
        ax.set_xticklabels(grid_sizes)
        ax.set_xlabel("Grid size [m]")
        if evaluation_metric == "Time":
            ax.yaxis.set_major_formatter(ScalarFormatterForceFormat_zero())
        elif evaluation_metric == "Distance":
            ax.yaxis.set_major_formatter(ScalarFormatterForceFormat_one())
        else:
            ax.yaxis.set_major_formatter(ScalarFormatterForceFormat_two())
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylabel(y_label)
        ax.set_title(evaluation_metric, fontsize=20)
        ax.legend(loc="lower right")

        plt.savefig(f"Result_{data_type}.png")
        plt.clf()
        plt.close()



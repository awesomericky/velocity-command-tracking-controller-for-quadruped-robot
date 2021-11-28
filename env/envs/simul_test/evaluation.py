import numpy as np
import matplotlib.pyplot as plt


## Compute "Comparison - Oracle" for Time and Distance

result_folder = "/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/Simple_point_goal_nav/Result"
grid_sizes = [2.5, 3.0]
standard = "Oracle"
comparisons = ["Naive", "CVAE", "CWM"]
quantile_percent = [25, 50, 75]

for comparison in comparisons:
    print("================================================")
    print(f"<<-- {comparison} -->>\n")
    for grid_size in grid_sizes:
        file = f"{str(grid_size)}_grid_result.npz"
        standard_data = np.load(f"{result_folder}/{standard}/{file}")
        standard_success = standard_data["success"]
        comparision_data = np.load(f"{result_folder}/{comparison}/{file}")
        comparison_success = comparision_data["success"]
        list_time_difference = []
        list_distance_difference = []

        assert len(standard_success) == len(comparison_success)
        n_total_data = len(standard_success)

        for i in range(n_total_data):
            # Just compare cases where both success
            if standard_success[i] and comparison_success[i]:
                standard_idx = np.sum(standard_success[:i+1]) - 1
                comparison_idx = np.sum(comparison_success[:i+1]) - 1

                list_time_difference.append(comparision_data["time"][comparison_idx] - standard_data["time"][standard_idx])
                list_distance_difference.append(comparision_data["distance"][comparison_idx] - standard_data["distance"][standard_idx])

        list_time_difference = np.array(list_time_difference)
        list_distance_difference = np.array(list_distance_difference)

        time_difference_mean = np.mean(list_time_difference)
        time_difference_quantile = np.percentile(list_time_difference, quantile_percent)
        distance_difference_mean = np.mean(list_distance_difference)
        distance_difference_quantile = np.percentile(list_distance_difference, quantile_percent)

        print(f"Grid_{str(grid_size)}:")
        print(f"Time difference: {round(time_difference_mean, 1)}  [{round(time_difference_quantile[0], 1)} / {round(time_difference_quantile[1], 1)} / {round(time_difference_quantile[2], 1)}]")
        print(f"Distance difference: {round(distance_difference_mean, 1)}  [{round(distance_difference_quantile[0], 1)} / {round(distance_difference_quantile[1], 1)} / {round(distance_difference_quantile[2], 1)}]")

        plt.scatter(range(len(list_time_difference)), list_time_difference, s=30)
        plt.savefig(f"{result_folder}/{comparison}_{str(grid_size)}_time.png")
        plt.clf()
        plt.scatter(range(len(list_distance_difference)), list_distance_difference, s=30)
        plt.savefig(f"{result_folder}/{comparison}_{str(grid_size)}_distance.png")
        plt.clf()


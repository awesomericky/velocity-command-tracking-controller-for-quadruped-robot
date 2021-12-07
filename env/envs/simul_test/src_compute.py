import numpy as np
import json
from collections import defaultdict

evaluate_seeds = [37, 143, 534]
grid_sizes = [2.5, 3.0, 4.0, 5.0]
evaluation_metrics = ["Collision safe", "No collision safe"]
# result_folder_path = "/home/awesomericky/Lab_intern/raisimLib/raisimGymTorch/Simple_safety_remote_control/Result"
result_folder_path = "/Users/kimyunho/Desktop/Simple_safety_remote_control/Result"

data_name = "Naive_1000"

# Generate empty text file to record data
open(f"src_result_{data_name}.txt", "w").close()

# Load recorded result & Compute mean and std
for i, grid_size in enumerate(grid_sizes):
    list_collision_safe = []
    list_no_collision_safe = []

    for evaluate_seed in evaluate_seeds:
        specific_folder_name = f"{data_name}_{str(evaluate_seed)}"
        file_name = f"{str(grid_size)}_grid_result.json"
        final_path = f"{result_folder_path}/{specific_folder_name}/{file_name}"
        with open(final_path, "r") as f:
            recorded_result = json.load(f)

        list_collision_safe.append(recorded_result["Collision"]["safe"])
        list_no_collision_safe.append(recorded_result["No collision"]["safe"])

    list_collision_safe = np.array(list_collision_safe)
    list_no_collision_safe = np.array(list_no_collision_safe)

    # Record result
    with open(f"src_result_{data_name}.txt", "a") as f:
        f.write(f"Grid_{str(grid_size)}:\n")
        f.write(f"Collision safe: {round(np.mean(list_collision_safe), 3)} ({round(np.std(list_collision_safe), 3)})\n")
        f.write(f"No collision safe: {round(np.mean(list_no_collision_safe), 3)} ({round(np.std(list_no_collision_safe), 3)})\n\n")
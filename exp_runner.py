import os
import subprocess
from datetime import datetime
from itertools import islice

import math

from session_handler.experiment import experiment_combinations, base_command, grid_keys, config
from session_handler.screen_sessions import get_idle_gpus, create_screen_session, list_screen_sessions

SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"src_path: {SRC_PATH}")

# Function to divide a large iterable into smaller parts
def split_experiment_combinations(combinations, n):
    """
    Splits the experiment_combinations into n approximately equal chunks.

    :param combinations: An iterable of experiment combinations (e.g., itertools.product object).
    :param n: Number of chunks to create (typically the number of GPUs).
    :return: List of n separate lists of combinations.
    """
    # Create a list of iterators
    iterators = [iter(combinations)] * n
    # Use islice to create approximately equal chunks
    chunk_size = math.ceil(len(combinations) / n)
    return [list(islice(it, chunk_size)) for it in iterators]


# Set maximum number of GPUs to use
max_gpu_to_use = 2  # Adjust this value as needed

# Detect idle GPUs
idle_gpus = get_idle_gpus()  # Replace with the actual function or hard-coded list like [0, 1, 2]
idle_gpus = idle_gpus[:min(max_gpu_to_use, len(idle_gpus))]
# Limit the number of GPUs used
num_idle_gpus = len(idle_gpus)

# Split experiment_combinations across idle GPUs
experiment_combinations_list = list(experiment_combinations)  # Convert to a list if it's an itertools.product

if num_idle_gpus > 0:
    # Divide combinations among available GPUs
    new_experiment_combinations = split_experiment_combinations(
        experiment_combinations_list, num_idle_gpus
    )

    # Display assignments
    for i, gpu_combinations in enumerate(new_experiment_combinations):
        print(f"GPU {idle_gpus[i]} receives {len(gpu_combinations)} combinations.")
        print(gpu_combinations)

else:
    print("No idle GPUs available.")


folder_name = f"{config['exp_name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
folder_path = os.path.join(SRC_PATH, "experiments", folder_name)
os.makedirs(folder_path, exist_ok=True)

print(f"folder_path: {folder_path}")

# Define include directories and files
include_dirs_files = [
    "src/***",  # Match all files and subdirectories recursively in `src`
    "envs/***", # Match all files and subdirectories recursively in `envs`
    "training.py",
    "utils.py"
]
include_opts = [f"--include={path}" for path in include_dirs_files] + ["--exclude=*", "--prune-empty-dirs"]

# Copy only specified files and directories to the folder path
rsync_command = ["rsync", "-av"] + include_opts + [f"{SRC_PATH}/", folder_path]  # Ensure trailing slash for SRC_PATH
print(f"rsync_command: {rsync_command}")

subprocess.run(rsync_command, check=True)


# Loop through each configuration and execute the command
for gpu, gpu_combinations in zip(idle_gpus, new_experiment_combinations):
    # Create one long command from gpu_combinations
    commands = []
    for combination in gpu_combinations:
        # Create a dictionary mapping keys (from grid) to their respective values
        combination_dict = dict(zip(grid_keys, combination))
        # Format the command with the combination values
        command = base_command.format(**combination_dict)
        commands.append(command)

    # Combine all commands with "&&" and prepend CUDA_VISIBLE_DEVICES
    full_command = f"CUDA_VISIBLE_DEVICES={gpu} " + f" && CUDA_VISIBLE_DEVICES={gpu} ".join(commands)
    print(f"Running on GPU {gpu}: {full_command}")
    create_screen_session(session_name=f"gpu_session_{gpu}", command=full_command)

    # List all screen sessions
    print("Active screen sessions:")
    list_screen_sessions()

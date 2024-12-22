from itertools import islice
import math

from session_handler.experiment import experiment_combinations
from session_handler.screen_sessions import get_idle_gpus


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


# Detect idle GPUs
idle_gpus = get_idle_gpus()  # Replace with the actual function or hard-coded list like [0, 1, 2]

# Split experiment_combinations across idle GPUs
experiment_combinations_list = list(experiment_combinations)  # Convert to a list if it's an itertools.product
num_idle_gpus = len(idle_gpus)

if num_idle_gpus > 0:
    # Divide combinations among available GPUs
    new_experiment_combinations = split_experiment_combinations(
        experiment_combinations_list, num_idle_gpus
    )

    # Display assignments
    for i, gpu_combinations in enumerate(new_experiment_combinations):
        print(f"GPU {idle_gpus[i]} receives {len(gpu_combinations)} combinations.")

else:
    print("No idle GPUs available.")
import itertools

# Configuration
config = {
    "exp_name": "crl",
    "project_name": "manipulation_new",
    "group_name": "first_run",
    "env": ["arm_reach", "arm_grasp", "arm_push_easy"],
    "n_hidden": [2, 3, 4],
    "h_dim": 1024,
    "seed": [2,3,4],
    "batch_size": [256],  # Example: Added to the grid search
    "num_envs": [256],  # Example: Added to the grid search
    "num_evals": 10,
    "num_timesteps": 100000000,
    "episode_length": 250,
    "unroll_length": 62,
    "min_replay_size": 1000,
    "contrastive_loss_fn": "symmetric_infonce",
    "max_replay_size": 10000,
    "discounting": 0.99,
    "action_repeat": 1,
    "repr_dim": 64,
    "multiplier_num_sgd_steps": 1,
    "energy_fn": "l2",
    "logsumexp_penalty": 0.1,
    "l2_penalty": 0.000001,
    "use_ln": True,
    "log_wandb": True,
    "disable_entropy_actor": True
}

# Identify ALL keys with lists to include in the grid search
grid_keys = [key for key, value in config.items() if isinstance(value, list)]

# Generate combinations dynamically for all grid search keys
grid_values = [config[key] for key in grid_keys]
experiment_combinations = itertools.product(*grid_values)

# Base command template (static arguments)
base_command = "python training.py " + " ".join(
    [
        f"--{key} {value}"
        for key, value in config.items()
        if not isinstance(value, list) and not isinstance(value, bool)  # Static arguments
    ]
) + " " + " ".join(
    [f"--{key}" for key, value in config.items() if isinstance(value, bool) and value]
)  # Booleans



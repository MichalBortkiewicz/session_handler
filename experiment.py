import itertools
import os
import subprocess
import tempfile

# Configuration
config = {
    "exp_name": "crl",
    "envs": ["arm_binpick_hard"],
    "hidden_layers": [2, 3, 4],
    "seeds": [7, 8, 9],
    "num_evals": 100,
    "project_name": "manipulation",
    "group_name": "first_run",
    "num_timesteps": 100000000,
    "batch_size": 256,
    "num_envs": 1024,
    "episode_length": 250,
    "unroll_length": 62,
    "min_replay_size": 1000,
    "contrastive_loss_fn": "symmetric_infonce",
    "max_replay_size": 10000,
    "discounting": 0.99,
    "action_repeat": 1,
    "h_dim": 1024,
    "repr_dim": 64,
    "multiplier_num_sgd_steps": 4,
    "energy_fn": "l2",
    "logsumexp_penalty": 0.1,
    "l2_penalty": 0.000001,
    "use_ln": True,
    "log_wandb": True,
    "disable_entropy_actor": True
}

# Create the main experiments directory if it doesn't exist
os.makedirs("./experiments", exist_ok=True)

# Define exclude directories separately
exclude_dirs = [
    "old_contrastive", "params", "renders", "scripts", "notebooks", "experiments", "wandb",
    "imgs", ".git", "__pycache__", "wykresy_crl", "plots", "clean_JaxGCRL"
]

# Create a temporary directory with the experiment name within ./experiments
temp_dir = tempfile.mkdtemp(prefix=f"{config['exp_name']}_", dir="./experiments")

# Create the rsync exclude options
exclude_opts = [f"--exclude={dir}" for dir in exclude_dirs]

# Copy all necessary files to the temporary directory, excluding specified directories
rsync_command = ["rsync", "-av"] + exclude_opts + ["./", temp_dir]
subprocess.run(rsync_command, check=True)

# Change to the temporary directory
os.chdir(temp_dir)
print(f"Current path: '{os.getcwd()}'")

# Activate the conda environment
subprocess.run(["conda", "shell.bash", "hook"], shell=True, check=True)
subprocess.run(["conda", "activate", "contrastive_rl"], shell=True, check=True)

# Generate all combinations of 'envs', 'hidden_layers', and 'seeds' using itertools.product
experiment_combinations = itertools.product(
    config['envs'],
    config['hidden_layers'],
    config['seeds']
)

# Base command template
base_command = (
    "python training.py "
    f"--num_evals {config['num_evals']} "
    f"--project_name {config['project_name']} "
    f"--group_name {config['group_name']} "
    f"--num_timesteps {config['num_timesteps']} "
    f"--batch_size {config['batch_size']} "
    f"--num_envs {config['num_envs']} "
    f"--exp_name {config['exp_name']} "
    f"--episode_length {config['episode_length']} "
    f"--unroll_length {config['unroll_length']} "
    f"--min_replay_size {config['min_replay_size']} "
    f"--contrastive_loss_fn '{config['contrastive_loss_fn']}' "
    f"--max_replay_size {config['max_replay_size']} "
    f"--discounting {config['discounting']} "
    f"--action_repeat {config['action_repeat']} "
    f"--h_dim {config['h_dim']} "
    f"--repr_dim {config['repr_dim']} "
    f"--multiplier_num_sgd_steps {config['multiplier_num_sgd_steps']} "
    f"--energy_fn '{config['energy_fn']}' "
    f"--logsumexp_penalty {config['logsumexp_penalty']} "
    f"--l2_penalty {config['l2_penalty']} "
    f"{'--use_ln' if config['use_ln'] else ''} "
    f"{'--log_wandb' if config['log_wandb'] else ''} "
    f"{'--disable_entropy_actor' if config['disable_entropy_actor'] else ''} "
    "--env_name {env} "
    "--n_hidden {hidden} "
    "--seed {seed}"
)

# Run experiments for each combination
for env, hidden, seed in experiment_combinations:
    command = base_command.format(env=env, hidden=hidden, seed=seed)
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)

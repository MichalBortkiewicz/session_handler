import subprocess


def create_screen_session(session_name, command=None):
    """
    Creates a new screen session with the specified name and optionally runs a command.

    :param session_name: Name of the screen session.
    :param command: Command to run in the session. If None, just creates the session.
    """
    try:
        # Start a new detached screen session
        if command:
            subprocess.run(
                ["screen", "-dmS", session_name, "bash", "-c", command],
                check=True
            )
        else:
            subprocess.run(
                ["screen", "-dmS", session_name],
                check=True
            )
        print(f"Screen session '{session_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating screen session: {e}")


def list_screen_sessions():
    """Lists all active screen sessions."""
    try:
        result = subprocess.run(
            ["screen", "-ls"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error listing screen sessions: {e}")


def attach_screen_session(session_name):
    """Attaches to an existing screen session."""
    try:
        subprocess.run(["screen", "-r", session_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error attaching to screen session: {e}")


def get_idle_gpus():
    """
    Detects idle GPUs using 'nvidia-smi' and returns a list of their IDs.

    :return: List of IDs of idle GPUs. If an error occurs, returns an empty list.
    """
    try:
        gpu_status = (
            subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        idle_gpus = [i for i, status in enumerate(gpu_status) if int(status) == 0]
        print(idle_gpus)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        idle_gpus = []
        print("Error detecting GPU status or no idle GPUs available. Exiting.")
        exit(1)
    return idle_gpus


# Example Usage
if __name__ == "__main__":

    # Identify GPUs that are idle
    idle_gpus = get_idle_gpus()
    print(idle_gpus)
    # Spawn a separate screen session for each idle GPU
    for gpu_id in idle_gpus:
        session_name = f"gpu_session_{gpu_id}"
        command_to_run = f"CUDA_VISIBLE_DEVICES={gpu_id} bash -c 'echo \"working\"; exec bash'"
        create_screen_session(session_name, command_to_run)

    # List all screen sessions
    print("Active screen sessions:")
    list_screen_sessions()

    
    # Attach to a specific session (optional, for interactive purposes)
    # attach_screen_session(f"gpu_session_{0}")  # Example for attaching to the first GPU session

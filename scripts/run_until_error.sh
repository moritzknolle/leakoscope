#!/bin/bash

# Check if a command was provided as an argument.
# '$#' is a special variable that holds the number of arguments passed to the script.
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# The command to run is all of the arguments passed to the script.
# "$@" represents all command-line arguments as separate words.
command_to_run="$@"

counter=0

echo "Starting to run command: $command_to_run"
echo "-----------------------------------"

# Start the while loop.
while $command_to_run; do
    counter=$((counter+1))
    echo "Command succesfully executed. Total successful runs: $counter"
    echo "-----------------------------------"
done

# sync wandb runs and delete local log directories
wandb sync --clean --clean-old-hours 1 --clean-force

# The loop has exited because the command failed.
echo "-----------------------------------"
echo "Command failed on attempt $((counter+1)). Exiting."
echo "Exit code of the last command: $?"
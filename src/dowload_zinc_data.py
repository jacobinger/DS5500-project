#!/usr/bin/env python3

import os
import subprocess

# Path to the command file and output directory
COMMAND_FILE = "ZINC-downloader-3D-smi.txt"
OUTPUT_DIR = "zinc_data"
SCRIPT_FILE = "download_zinc.sh"  # Changed to .sh

def create_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        for file in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, file))

def generate_bash_script():
    """Generate a Bash script from the curl commands."""
    with open(COMMAND_FILE, "r") as f:
        commands = f.readlines()

    wget_commands = []
    for cmd in commands:
        cmd = cmd.strip()
        if cmd and cmd.startswith("curl"):
            parts = cmd.split()
            url = parts[-1]
            output_path = parts[parts.index("-o") + 1]
            relative_path = os.path.join(OUTPUT_DIR, os.path.relpath(output_path, start=os.getcwd()))
            wget_cmd = f"wget -P {os.path.dirname(relative_path)} {url}"
            wget_commands.append(wget_cmd)

    with open(SCRIPT_FILE, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Generated Bash script to download ZINC data\n")
        f.write("set -e  # Exit on error\n")
        for cmd in wget_commands:
            f.write(f"{cmd}\n")
        f.write("echo 'Download complete!'\n")

    os.chmod(SCRIPT_FILE, 0o755)

def run_bash_script():
    """Run the generated Bash script."""
    try:
        subprocess.run(["bash", SCRIPT_FILE], check=True)
        print(f"Successfully downloaded files to {OUTPUT_DIR}/")
    except subprocess.CalledProcessError as e:
        print(f"Error running bash script: {e}")
        exit(1)

def main():
    print("Setting up ZINC data download...")
    create_output_directory()
    generate_bash_script()
    run_bash_script()

if __name__ == "__main__":
    main()
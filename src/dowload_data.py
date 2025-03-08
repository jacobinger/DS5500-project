#!/usr/bin/env python3

import os
import subprocess

# Path to the command file and output directory
COMMAND_FILE = "ZINC-downloader-3D-smi.txt"
OUTPUT_DIR = "zinc_data"
CSCRIPT_FILE = "download_zinc.csh"

def create_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        # Clear the directory to avoid duplicates
        for file in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, file))

def generate_csh_script():
    """Generate a C-shell script from the curl commands."""
    with open(COMMAND_FILE, "r") as f:
        commands = f.readlines()

    # Convert curl to wget commands (csh-compatible)
    wget_commands = []
    for cmd in commands:
        cmd = cmd.strip()
        if cmd and cmd.startswith("curl"):
            # Extract URL and output path
            parts = cmd.split()
            url = parts[-1]  # Last part is the URL
            output_path = parts[parts.index("-o") + 1]  # Get the -o argument
            # Adjust output path to be relative to OUTPUT_DIR
            relative_path = os.path.join(OUTPUT_DIR, os.path.relpath(output_path, start=os.getcwd()))
            # Convert to wget (csh syntax uses set and wget)
            wget_cmd = f"wget -P {os.path.dirname(relative_path)} {url}"
            wget_commands.append(wget_cmd)

    # Write the C-shell script
    with open(CSCRIPT_FILE, "w") as f:
        f.write("#!/bin/csh\n")
        f.write("# Generated C-shell script to download ZINC data\n")
        f.write("set -e  # Exit on error\n")
        for cmd in wget_commands:
            f.write(f"{cmd}\n")
        f.write("echo 'Download complete!'\n")

    # Make the script executable
    os.chmod(CSCRIPT_FILE, 0o755)

def run_csh_script():
    """Run the generated C-shell script."""
    try:
        subprocess.run(["csh", CSCRIPT_FILE], check=True)
        print(f"Successfully downloaded files to {OUTPUT_DIR}/")
    except subprocess.CalledProcessError as e:
        print(f"Error running csh script: {e}")
        exit(1)

def main():
    print("Setting up ZINC data download...")
    create_output_directory()
    generate_csh_script()
    run_csh_script()

if __name__ == "__main__":
    main()
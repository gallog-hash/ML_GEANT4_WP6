#!/usr/bin/env python
"""
Run vae_generate.py with different downsample/upsample factors.

This script automates running vae_generate.py across multiple upsampling
factors for systematic comparison and analysis.
"""

import json
import subprocess
import sys
from pathlib import Path

# Factors to test
FACTORS = [10, 20, 50, 100]

# Configuration files (relative to project root)
CONFIG_FILE = Path("src/configs/generation_profiles.json")
BACKUP_FILE = Path("src/configs/generation_profiles.json.backup")


def update_profile_factors(factor):
    """Update both downsample_factor and upsample_factor in profile."""
    with open(CONFIG_FILE, "r") as f:
        profiles = json.load(f)

    # Update downsample profile
    profiles["downsample"]["downsample_factor"] = factor
    profiles["downsample"]["upsample_factor"] = factor

    with open(CONFIG_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Updated profile: downsample_factor={factor}, upsample_factor={factor}")


def main():
    # Get the directory where this script is located (src/)
    script_dir = Path(__file__).resolve().parent

    # Project root is parent of src/
    project_root = script_dir.parent

    # Change to project root for consistent path resolution
    import os

    original_dir = os.getcwd()
    os.chdir(project_root)

    try:
        # Create output directory if it doesn't exist
        output_dir = Path("vae_generate_output")
        output_dir.mkdir(exist_ok=True)

        # Backup original config
        with open(CONFIG_FILE, "r") as f:
            original_config = f.read()

        with open(BACKUP_FILE, "w") as f:
            f.write(original_config)

        print("Backed up original configuration")

        try:
            for factor in FACTORS:
                print("\n" + "=" * 60)
                print(f"Running with factor: {factor}")
                print("=" * 60)

                # Update configuration
                update_profile_factors(factor)

                # Run generation - save output to vae_generate_output
                # directory
                output_file = output_dir / f"vae_generate_factor_{factor}.txt"
                with open(output_file, "w") as f:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "src/vae_generate.py",
                            "--config_path",
                            "src/configs/generation_config.json",
                            "--profile",
                            "downsample",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                    # Write to file and print to console
                    f.write(result.stdout)
                    print(result.stdout)

                print(f"\nCompleted factor {factor}")
                print(f"Output saved to: {output_file}")

        finally:
            # Restore original config
            with open(BACKUP_FILE, "r") as f:
                original_config = f.read()

            with open(CONFIG_FILE, "w") as f:
                f.write(original_config)

            print("\nRestored original configuration")
            print("All runs completed!")

    finally:
        # Restore original working directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

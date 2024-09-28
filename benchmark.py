import json
import os
import subprocess

import numpy as np


def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def run_experiment(config, num_runs=10):
    results = []
    output_dir = config["folder_name"]
    config.pop("folder_name")
    for i in range(num_runs):

        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/output_{i}.json"

        command = f"python {config['script_path']} "
        command += " ".join(
            [f"--{k}={v}" for k, v in config.items() if k != "script_path"]
        )
        command += f" --output_file={output_file}"

        print(f"Running command: {command}")

        subprocess.run(command, shell=True)

        with open(output_file, "r") as f:
            result_value = json.load(f)

        results.append(result_value)

    return results


def main(config_file, num_runs):
    config = load_config(config_file)
    config["folder_name"] = config_file.split("/")[-1].replace(".json", "")

    results = run_experiment(config, num_runs)

    test_metrics = {}
    for result in results:
        for key, value in result["test_metrics"][0].items():
            if key not in test_metrics:
                test_metrics[key] = []
            test_metrics[key].append(value)
        if "total_elapsed_time" not in test_metrics:
            test_metrics["total_elapsed_time"] = []
        test_metrics["total_elapsed_time"].append(result["total_elapsed_time"])

    summary = {}
    for key, value in test_metrics.items():
        mean = np.mean(value)
        std = np.std(value)
        summary[key] = {"mean": mean, "std": std}

    print("Test summary:", summary)

    with open(f"{config_file.split("/")[-1].replace(".json", "")}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ML experiment multiple times and calculate mean and std."
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the JSON configuration file"
    )
    parser.add_argument(
        "--num_runs", type=int, default=10, help="Number of times to run the experiment"
    )

    args = parser.parse_args()

    main(args.config_file, args.num_runs)

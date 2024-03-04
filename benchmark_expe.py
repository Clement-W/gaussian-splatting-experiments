import os
import yaml
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(description="Benchmark script parameters")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--output_path", default="../output")
    parser.add_argument("--expe_config", default=None)
    parser.add_argument("--dataset_path", default="../data_expe")
    return parser.parse_args()

def run_command(command):
    print("------------------------")
    print("Calling: " + command)
    os.system(command)

def run_experiment(expe_arg, flag, scenes, args):
    common_args = "--eval -r 2" if not args.skip_training else ""

    for scene in scenes:
        source = f"{args.dataset_path}/{scene}"
        output = f"{args.output_path}/{scene}/{flag}"
        
        if not args.skip_training:
            train_cmd = f"python train.py -s {source} -m {output} {expe_arg} {common_args}"
            run_command(train_cmd)
        
        if not args.skip_rendering:
            render_cmd = f"python render.py -s {source} -m {output}"
            run_command(render_cmd)
        
        if not args.skip_metrics:
            metrics_cmd = f"python metrics.py -m {output}"
            run_command(metrics_cmd)

def run_experiments_from_config(config_path, scenes, args):
    with open(config_path, 'r') as stream:
        params = yaml.safe_load(stream)['params']
    
    for param, values in params.items():
        for val in values:
            expe_arg = f"--{param} {val}"
            flag = f"{param}_{val}"
            run_experiment(expe_arg, flag, scenes, args)

def main():
    args = parse_arguments()
    #scenes = ["truck", "train", "playroom"]
    scenes = ["playroom"]

    if args.expe_config is None:
        print("Running baseline experiment")
        run_experiment("", "baseline", scenes, args)
    else:
        print(f"Running experiments from config {args.expe_config}")
        run_experiments_from_config(args.expe_config, scenes, args)

if __name__ == "__main__":
    main()
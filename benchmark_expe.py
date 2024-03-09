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
    parser.add_argument("--grid_search_regularization", default=None)
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
            common_args += " -w" if scene == "raindeer" else ""
            train_cmd = f"python train.py -s {source} -m {output} {expe_arg} {common_args}"
            run_command(train_cmd)
        
        if not args.skip_rendering:
            render_cmd = f"python render.py -s {source} -m {output} --skip_train"
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

def run_custom_gridsearch(scenes, args):

    methods = { # tout, juste le début pour contraindre, toute la partie densification, milieu/fin densification, postdensification
        "maxvariance_regularization": [[500, 30000], [500, 7000], [500, 15000], [7000, 15000], [15000, 28000]],
        "opacity_regularization": [[7000, 15000], [15000, 28000], [2500, 30000]],
        "edge_regularization": [[500, 30000], [500, 7000], [500, 15000], [7000, 15000], [15000, 28000]],
        "smoothness_regularization": [[500, 30000], [500, 7000], [500, 15000], [7000, 15000], [15000, 28000]]
    }
    lambdas = {
        "maxvariance_regularization":0.00001,
        "opacity_regularization":0.000001,
        "edge_regularization":0.001,
        "smoothness_regularization":0.00001
    }

    for method, intervals in methods.items():
        l = lambdas[method]
        for interval in intervals:
            expe_arg = f"--regularization_type {method} --lambda_regularization {l} --regularize_from_iter {interval[0]} --regularize_until_iter {interval[1]}"
            flag = f"{method}_{l}_[{interval[0]},{interval[1]}]"
            run_experiment(expe_arg, flag, scenes, args)

def run_custom_gridsearch2(scenes, args):

    methods = ["maxvariance_regularization", "opacity_regularization", "edge_regularization", "smoothness_regularization"]
    #combinations = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [0,1,2,3], [0,1,2], [0,1,3], [0,2,3], [1,2,3]] # indices of methods to combine
    #combinations = [[1,3], [1,2], [0,1,2], [0,2,3], [2], [0], [3]] 
    combinations = [[1,2], [0,2,3], [2], [3]] 
    lambdas = {
        "maxvariance_regularization":0.001,
        "opacity_regularization":0.0000001,
        "edge_regularization":0.01,
        "smoothness_regularization":0.001
    }

    for combination in combinations:
        methods_to_combine = [methods[i] for i in combination]
        expe_args=""
        for method in methods_to_combine:
            expe_args += f"--{method} {lambdas[method]} "

        flag = "_".join(methods_to_combine)
        run_experiment(expe_args, flag, scenes, args)
    
def main():
    args = parse_arguments()
    #scenes = ["truck", "train", "playroom","raindeer"]
    scenes = ["raindeer"]

    if args.expe_config is None and args.grid_search_regularization is None:
        print("Running baseline experiment")
        run_experiment("", "baseline", scenes, args)
    elif args.grid_search_regularization is not None:
        print("Running grid search for regularization")
        run_custom_gridsearch2(scenes,args)
    else:
        print(f"Running experiments from config {args.expe_config}")
        run_experiments_from_config(args.expe_config, scenes, args)

if __name__ == "__main__":
    main()
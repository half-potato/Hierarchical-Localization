import pipelines

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Run a method on a pipeline")
    parser.add_argument("base_dir", type=str, help='Path to directory containing all datasets')
    parser.add_argument("output_dir", type=str, help='Path to directory to store results')
    parser.add_argument("method", type=str, help='Name of model to run. Use "all" to run on all methods')
    parser.add_argument("pipeline", type=str, help='Name of pipeline to run model on. Use "all" to run on all pipelines')
    parser.add_argument("--run_localization", default=False, action="store_true", help='Whether to run the localization necessary to submit a benchmark to visuallocalization.net')
    args = parser.parse_args()
    if args.pipeline == "all":
        to_run = list(pipelines.PIPELINES.keys())
    else:
        to_run = [args.pipeline]
    if args.method == "all":
        methods = pipelines.METHODS
    else:
        methods = [pipelines.get_config(args.method)]

    for pipeline_name in to_run:
        for config in methods:
            print(f"Running {config['name']} on {pipeline_name}")
            pipelines.run_pipeline(args.base_dir, args.output_dir, pipeline_name, config, run_localization=args.run_localization)

#!/usr/bin/env python3
"""Entry point to training/validating/testing for a user given experiment
name."""
import os, sys, json

from typing import Dict, Tuple

import hydra
import prior
import importlib
import inspect
import ast
from omegaconf import DictConfig

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner, CONFIG_KWARGS_STR
from allenact.main import load_config, _config_source, find_sub_modules
from allenact.utils.system import get_logger

if "PROCTHOR_HYDRA_CONFIG_DIR" not in os.environ:
    os.environ["PROCTHOR_HYDRA_CONFIG_DIR"] = os.path.join(os.getcwd(), "config")
else:
    os.environ["PROCTHOR_HYDRA_CONFIG_DIR"] = os.path.abspath(
        os.environ["PROCTHOR_HYDRA_CONFIG_DIR"]
    )


def init_config(cfg: DictConfig) -> DictConfig:
    print(cfg)

    # NOTE: Support loading in model from prior
    allenact_checkpoint = None
    if cfg.checkpoint is not None and cfg.pretrained_model.name is not None:
        raise ValueError(
            f"Cannot specify both checkpoint {cfg.checkpoint}"
            f" and prior_checkpoint {cfg.pretrained_model.name}"
        )
    elif cfg.checkpoint is None and cfg.pretrained_model.name is not None:
        cfg.checkpoint = prior.load_model(
            project=cfg.pretrained_model.project, model=cfg.pretrained_model.name
        )

    return cfg


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, str]]:
    assert os.path.exists(
        args.experiment_base
    ), "The path '{}' does not seem to exist (your current working directory is '{}').".format(
        args.experiment_base, os.getcwd()
    )
    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(args.experiment_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = args.experiment
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            f"Could not import experiment '{module_path}', are you sure this is the right path?"
            f" Possibly relevant files include {relevant_submodules}."
            f" Note that the experiment must be reachable along your `PYTHONPATH`, it might"
            f" be helpful for you to run `export PYTHONPATH=$PYTHONPATH:$PWD` in your"
            f" project's top level directory."
        ) from e

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config_kwargs = {}
    if args.config_kwargs is not None:
        if os.path.exists(args.config_kwargs):
            with open(args.config_kwargs, "r") as f:
                config_kwargs = json.load(f)
        else:
            try:
                config_kwargs = json.loads(args.config_kwargs)
            except json.JSONDecodeError:
                get_logger().warning(
                    f"The input for --config_kwargs ('{args.config_kwargs}')"
                    f" does not appear to be valid json. Often this is due to"
                    f" json requiring very specific syntax (e.g. double quoted strings)"
                    f" we'll try to get around this by evaluating with `ast.literal_eval`"
                    f" (a safer version of the standard `eval` function)."
                )
                config_kwargs = ast.literal_eval(args.config_kwargs)

        assert isinstance(
            config_kwargs, Dict
        ), "`--config_kwargs` must be a json string (or a path to a .json file) that evaluates to a dictionary."

    config = experiments[0](cfg=args, **config_kwargs)
    sources = _config_source(config_type=experiments[0])
    sources[CONFIG_KWARGS_STR] = json.dumps(config_kwargs)
    return config, sources


@hydra.main(config_path=os.environ["PROCTHOR_HYDRA_CONFIG_DIR"], config_name="main")
def main(cfg: DictConfig) -> None:
    cfg = init_config(cfg=cfg)

    exp_cfg, srcs = load_config(cfg)
    runner = OnPolicyRunner(
        config=exp_cfg,
        output_dir=cfg.output_dir,
        loaded_config_src_files=srcs,
        seed=cfg.seed,
        disable_tensorboard=cfg.disable_tensorboard,
        callbacks_paths=cfg.callbacks,
        mode="test" if cfg.eval else "train",
        machine_id=cfg.distributed.machine_id,
        distributed_ip_and_port=cfg.distributed.ip_and_port,
        extra_tag=cfg.extra_tag,
    )
    if cfg.eval:
        runner.start_test(checkpoint_path_dir_or_pattern=cfg.checkpoint)
    else:
        runner.start_train(
            checkpoint=cfg.checkpoint,
            valid_on_initial_weights=cfg.valid_on_initial_weights,
            try_restart_after_task_error=cfg.enable_crash_recovery,
            restart_pipeline=cfg.restart_pipeline,
        )


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=./")
    main()

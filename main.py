from urllib.parse import urlparse

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import (
    load_libsvm_dataset,
    create_data_loaders,
    FixLength,
)
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import (
    create_output_dirs,
    PathsContainer,
    copy_local_to_gs,
)
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
import datetime

# python main.py --job_dir haolun_pointwise --config_file_name scripts/config_haolun_pointwise.json
# python main.py --job_dir haolun_pairwise --config_file_name scripts/config_haolun_pairwise.json


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument(
        "--job_dir", help="Base output path for all experiments", required=True
    )
    parser.add_argument(
        "--run_id",
        help="Name of this run to be recorded (must be unique within output dir)",
        # required=True,
        default="play",
    )
    parser.add_argument(
        "--config_file_name",
        required=True,
        type=str,
        help="Name of json file with config",
    )

    return parser.parse_args()


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    # # Get the current time
    # current_time = datetime.datetime.now()
    # args.run_id = current_time.strftime("%Y-%m-%d-%H-%M")

    paths = PathsContainer.from_args(
        "job_dir/" + args.job_dir, args.run_id, args.config_file_name
    )

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    # train_ds: [n_train_query, max_train_query_doc_length, n_feature]
    # later will be padded to: [batch_size, slate_length, n_feature]
    # val_ds: [n_val_query, max_val_query_doc_length, n_feature]
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )
    print("train_ds:", train_ds.shape, train_ds)
    print("val_ds:", val_ds.shape, val_ds)

    n_features = train_ds.shape[-1]
    assert (
        n_features == val_ds.shape[-1]
    ), "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds,
        val_ds,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
    )

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info(
            "Model training will be distributed to {} GPUs.".format(
                torch.cuda.device_count()
            )
        )
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(
        params=model.parameters(), **config.optimizer.args
    )
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(
            optimizer, **config.lr_scheduler.args
        )
    else:
        scheduler = None

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        result = fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            **asdict(config.training),
        )

    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)


if __name__ == "__main__":
    run()

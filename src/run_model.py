import sys
import time
import traceback
import torch
import yaml
import argparse
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add src directory to Python path
module_path = Path("src").resolve().as_posix()
print(module_path)
sys.path.insert(0, module_path)

# Import custom modules
try:
    from custom_dataset import CropData
    import utils as u
    from models.unet import Unet
    from model_compiler import ModelCompiler
    from custom_loss_functions import *
except ModuleNotFoundError:
    print("Module not found")
    pass


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            print(f"Function '{func.__name__}' failed after {end_time - start_time:.4f} seconds")
            print(f"Error: {e}")
            traceback.print_exc()
            raise e
    return wrapper


def load_config(yaml_config_path, num_time_points):
    print(yaml_config_path)
    with open(yaml_config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    for key, value in config["global_stats"].items():
        config["global_stats"][key] = value * num_time_points

    keys_with_paths = [
        "dataset_path",
        "train_csv_path",
        "inference_csv_path",
        "validation_csv_path",
        "tensorboard_log_path",
    ]
    for key in keys_with_paths:
        config[key] = Path(config[key]).resolve().as_posix()

    if config["params_init"]:
        config["params_init"] = Path(config["params_init"]).resolve().as_posix()

    # Initialize the SummaryWriter by setting the directory where to save the event files
    # The events.out.tfevents files are created by TensorBoard to log events from all training events
    writer = SummaryWriter(log_dir=config["tensorboard_log_path"])

    return config

@timeit
def prepare_data(config, usage):
    dataset = CropData(
        src_dir=config["dataset_path"],
        usage=usage,
        dataset_name=config["dataset_dir"],
        csv_path=config[f"{usage}_csv_path"],
        apply_normalization=config["apply_normalization"],
        normal_strategy=config["normal_strategy"],
        stat_procedure=config["stat_procedure"],
        global_stats=config["global_stats"],
        trans=config["transformations"] if usage == "train" else None,
        **config["aug_params"] if usage == "train" else {},
    )
    return dataset


def create_dataloader(dataset, batch_size, shuffle, collate_fn=None):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def initialize_model(config):
    return Unet(
        n_classes=config["n_classes"],
        in_channels=config["input_channels"],
        use_skipAtt=config["use_skipAtt"],
        filter_config=config["filter_config"],
        dropout_rate=config["train_dropout_rate"],
    )


def compile_model(model, config):
    return ModelCompiler(
        model,
        working_dir=config["working_dir"],
        out_dir=config["out_dir"],
        num_classes=config["n_classes"],
        inch=config["input_channels"],
        class_mapping=config["class_mapping"],
        gpu_devices=config["gpuDevices"],
        model_init_type=config["init_type"],
        params_init=config["params_init"],
        freeze_params=config["freeze_params"],
    )


@timeit
def train_model(compiled_model, train_loader, val_loader, config):
    criterion_name = config["criterion"]["name"]
    weight = config["criterion"]["weight"]
    ignore_index = config["criterion"]["ignore_index"]

    if criterion_name == "TverskyFocalLoss":
        criterion = TverskyFocalLoss(
            weight=weight, ignore_index=ignore_index, gamma=config["criterion"]["gamma"]
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    compiled_model.fit(
        train_loader,
        val_loader,
        epochs=config["epochs"],
        optimizer_name=config["optimizer"],
        lr_init=config["LR"],
        lr_policy=config["LR_policy"],
        criterion=criterion,
        momentum=config["momentum"],
        checkpoint_interval=config["checkpoint_interval"],
        resume=config["resume"],
        resume_epoch=config["resume_epoch"],
        **config["lr_prams"],
    )


def save_model(compiled_model):
    compiled_model.save(save_object="params")


def evaluate_model(compiled_model, val_loader, config):
    compiled_model.accuracy_evaluation(val_loader, filename=config["val_metric_fname"])


def inference_model(compiled_model, test_loader):
    compiled_model.inference(test_loader)


def meta_handling_collate_fn(batch):
    images, labels, img_ids, img_metas = [], [], [], []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
        img_ids.append(sample[2])
        img_metas.append(sample[3])
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels, img_ids, img_metas

@timeit
def main(yaml_config_path, num_time_points, mode):
    config = load_config(yaml_config_path, num_time_points)

    if mode == "train":
        train_dataset = prepare_data(config, "train")
        val_dataset = prepare_data(config, "validation")

        train_loader = create_dataloader(
            train_dataset, config["train_BatchSize"], shuffle=True
        )
        val_loader = create_dataloader(
            val_dataset, config["val_test_BatchSize"], shuffle=False
        )

        model = initialize_model(config)
        compiled_model = compile_model(model, config)

        train_model(compiled_model, train_loader, val_loader, config)
        save_model(compiled_model)
        evaluate_model(compiled_model, val_loader, config)

    elif mode == "validate":
        val_dataset = prepare_data(config, "validation")
        val_loader = create_dataloader(
            val_dataset, config["val_test_BatchSize"], shuffle=False
        )

        model = initialize_model(config)
        compiled_model = compile_model(model, config)

        evaluate_model(compiled_model, val_loader, config)

    elif mode == "inference":
        test_dataset = prepare_data(config, "inference")
        test_loader = create_dataloader(
            test_dataset,
            config["val_test_BatchSize"],
            shuffle=False,
            collate_fn=meta_handling_collate_fn,
        )

        model = initialize_model(config)
        compiled_model = compile_model(model, config)

        inference_model(compiled_model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model pipeline")
    parser.add_argument(
        "mode",
        choices=["train", "validate", "inference"],
        help="Mode to run the script in",
    )
    parser.add_argument(
        "--config",
        default=("./config/default_config.yaml"),
        help="Path to the config file",
    )
    parser.add_argument(
        "--num_time_points", type=int, default=3, help="Number of time points"
    )

    args = parser.parse_args()
    yaml_config_path = Path(args.config).resolve()

    main(yaml_config_path, args.num_time_points, args.mode)

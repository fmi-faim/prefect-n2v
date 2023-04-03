import json
import os
from os.path import exists, join

from cpr.numpy.NumpySource import NumpySource
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from faim_prefect.mamba import log_infrastructure
from faim_prefect.parameter import User
from prefect import flow, get_run_logger, task
from prefect.filesystems import LocalFileSystem

from flows.storage_keys import RESULT_STORAGE_KEY
from flows.tasks.train import train_model
from flows.utils.parameters import N2VModel, TrainData, WandB


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def validate_parameters(
    user: User,
    run_name: str,
    train_data: TrainData,
    n2v_model: N2VModel,
    wandb: WandB,
):
    base_dir = LocalFileSystem.load("base-output-directory").basepath
    group = user.group.value
    assert exists(join(base_dir, group)), (
        f"Group '{group}' does not exist " f"in '{base_dir}'."
    )

    assert exists(
        train_data.train_data
    ), f"Train data '{train_data.train_data}' does not exist."

    assert exists(
        train_data.val_data
    ), f"Train data '{train_data.val_data}' does not exist."

    assert n2v_model.epochs >= 1, "Number of epochs must be >= 1."
    assert n2v_model.batch_size >= 1, "Batch size must be >= 1."
    assert n2v_model.unet_depth >= 1, "unet_depth must be >= 1."

    parameters = {
        "user": {
            "name": user.name,
            "group": group,
        },
        "run_name": run_name,
        "train_data": train_data.dict(),
        "n2v_model": n2v_model.dict(),
        "wandb": wandb.dict(),
    }

    run_dir = join(
        base_dir, group, user.name, "prefect-runs", "n2v", run_name.replace(" ", "-")
    )

    assert not exists(run_dir), f"Run directory {run_dir} exists already."

    os.makedirs(run_dir, exist_ok=False)
    with open(join(run_dir, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters, indent=4))

    return run_dir


with open(
    join("flows/train_n2v_2D.md"),
    encoding="UTF-8",
) as f:
    description = f.read()


@flow(
    name="N2V: Train 2D model",
    description=description,
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("prefect-n2v"),
)
def train_n2v_2d(
    user: User,
    run_name: str,
    train_data: TrainData = TrainData(),
    n2v_model: N2VModel = N2VModel(),
    wandb: WandB = WandB(),
):
    run_dir = validate_parameters(
        user=user,
        run_name=run_name,
        train_data=train_data,
        n2v_model=n2v_model,
        wandb=wandb,
    )

    logger = get_run_logger()
    logger.info(f"Run logs are written to: {run_dir}")
    logger.info(
        f"N2V model '{n2v_model.model_name}' is saved in: " f"{n2v_model.output_dir}"
    )

    train_task = train_model(
        user=user,
        n2v_model=n2v_model,
        wandb_options=wandb,
        x=NumpySource.from_path(train_data.train_data),
        x_val=NumpySource.from_path(train_data.val_data),
        wait_for=[run_dir],
    )

    log_infrastructure(run_dir, wait_for=[train_task])

    return join(n2v_model.output_dir, n2v_model.model_name)

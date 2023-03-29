from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.utilities.utilities import task_input_hash
from faim_prefect.parameter import User
from n2v.models import N2V, N2VConfig
from prefect import get_run_logger, task
from prefect.blocks.system import Secret
from wandb.integration.keras import WandbCallback

import wandb
from flows.utils.parameters import N2VModel, WandB


@task(cache_key_fn=task_input_hash)
def train_model(
    user: User,
    n2v_model: N2VModel,
    wandb_options: WandB,
    x: NumpyTarget,
    x_val: NumpyTarget,
):
    wandb.login(
        key=Secret.load(wandb_options.secret_key).get(),
        host=wandb_options.host,
    )
    wandb.init(
        project=wandb_options.project,
        entity=wandb_options.entity,
        name=n2v_model.model_name,
        tags=[user.name, user.group.value],
    )

    get_run_logger().info(
        f"You can follow the training process at:\n" f"{wandb.run.get_url()}"
    )

    X = x.get_data()
    config = N2VConfig(
        X,
        unet_n_depth=n2v_model.unet_depth,
        unet_kern_size=3,
        unet_residual=False,
        train_steps_per_epoch=int(X.shape[0] / n2v_model.batch_size),
        train_epochs=n2v_model.epochs,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=n2v_model.batch_size,
        n2v_perc_pix=0.198,
        n2v_patch_shape=(64, 64),
        n2v_manipulator="median",
        blurpool=True,
        skip_skipone=True,
        n2v_neighborhood_radius=2,
    )
    model = N2V(
        config=config,
        name=n2v_model.model_name,
        basedir=n2v_model.output_dir,
    )
    model.prepare_for_training(metrics=())
    model.callbacks.append(WandbCallback())

    model.train(X, x_val.get_data())

    wandb.finish()

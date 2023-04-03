import json
import os
import re
import threading
from os.path import exists, join

from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from faim_prefect.mamba import log_infrastructure
from faim_prefect.parallelization.utils import wait_for_task_run
from faim_prefect.parameter import User
from n2v.models import N2V
from prefect import flow, get_run_logger, task
from prefect.filesystems import LocalFileSystem

from flows.generate_train_data_2DTime_to_2D import list_images
from flows.storage_keys import RESULT_STORAGE_KEY
from flows.tasks.predict import predict_2d
from flows.utils.parameters import InputData, N2VTrainedModel


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def validate_parameters(
    user: User,
    run_name: str,
    input_data: InputData,
    n2v_trained_model: N2VTrainedModel,
    output_dir: str,
):
    base_dir = LocalFileSystem.load("base-output-directory").basepath
    group = user.group.value
    assert exists(join(base_dir, group)), (
        f"Group '{group}' does not exist " f"in '{base_dir}'."
    )

    assert exists(input_data.input_dir), (
        f"Input directory " f"'{input_data.input_dir}' does not " f"exist."
    )

    assert bool(
        re.match("[TXYC]+", input_data.axes)
    ), "Axes is only allowed to contain 'TYXC'."

    model_dir = join(n2v_trained_model.base_dir, n2v_trained_model.model_name)
    assert exists(model_dir), f"N2V model '{model_dir}' does not exist."

    assert not exists(output_dir), f"Output directory {output_dir} exists " f"already."

    os.makedirs(output_dir, exist_ok=False)

    parameters = {
        "user": {
            "name": user.name,
            "group": group,
        },
        "run_name": run_name,
        "output_dir": output_dir,
        "n2v_trained_model": n2v_trained_model.dict(),
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
    join("flows/predict_n2v_2D.md"),
    encoding="UTF-8",
) as f:
    description = f.read()


@flow(
    name="N2V: Predict 2D model",
    description=description,
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("prefect-n2v"),
)
def predict_n2v_2d(
    user: User,
    run_name: str,
    input_data: InputData = InputData(),
    output_dir: str = "/tungstenfs/scratch",
    n2v_trained_model: N2VTrainedModel = N2VTrainedModel(),
):
    run_dir = validate_parameters(
        user=user,
        run_name=run_name,
        input_data=input_data,
        n2v_trained_model=n2v_trained_model,
        output_dir=output_dir,
    )

    logger = get_run_logger()
    logger.info(f"Run logs are written to: {run_dir}")
    logger.info(f"Predictions are saved in {output_dir}.")

    images = list_images(
        input_dir=input_data.input_dir,
        pattern=input_data.pattern,
        pixel_resolution_um=input_data.xy_pixelsize_um,
        axes=input_data.axes,
    )

    model = N2V(
        config=None,
        name=n2v_trained_model.model_name,
        basedir=n2v_trained_model.base_dir,
    )
    model.load_weights(f"weights_{n2v_trained_model.weights}.h5")

    gpu_semaphore = threading.Semaphore(1)

    buffer = []
    predictions = []
    for img in images:
        buffer.append(
            predict_2d.submit(
                model=model,
                img=img,
                output_dir=output_dir,
                gpu_semaphore=gpu_semaphore,
            )
        )

        wait_for_task_run(
            results=predictions,
            buffer=buffer,
            max_buffer_length=2,
            result_insert_fn=lambda r: r.result(),
        )

    wait_for_task_run(
        results=predictions,
        buffer=buffer,
        max_buffer_length=0,
        result_insert_fn=lambda r: r.result(),
    )

    log_infrastructure(run_dir)

    return predictions

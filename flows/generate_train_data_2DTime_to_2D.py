import json
import os
import re
from os.path import exists, join

from cpr.image.ImageSource import ImageSource
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from faim_prefect.mamba import log_infrastructure
from faim_prefect.parameter import User
from prefect import flow, get_run_logger, task
from prefect.filesystems import LocalFileSystem

from flows.tasks.data_generation import extract_patches
from flows.utils.parameters import DataGen2D, InputData


@task(cache_key_fn=task_input_hash)
def validate_parameters(
    user: User,
    run_name: str,
    input_data: InputData,
    output_dir: str,
    datagen_2d: DataGen2D,
):
    logger = get_run_logger()
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

    assert (
        len(datagen_2d.patch_shape) == 2
    ), "datagen_2d.patch_shape must be of length 2."

    assert not exists(output_dir), f"Output directory {output_dir} exists " f"already."
    os.makedirs(output_dir, exist_ok=False)

    run_dir = join(
        base_dir, group, user.name, "prefect-runs", "n2v", run_name.replace(" ", "-")
    )

    if exists(run_dir):
        logger.error(f"Run directory {run_dir} exists already.")

    parameters = {
        "user": {
            "name": user.name,
            "group": group,
        },
        "run_name": run_name,
        "input_data": input_data.dict(),
        "output_dir": output_dir,
        "datagen_2d": datagen_2d.dict(),
    }

    os.makedirs(run_dir, exist_ok=False)
    with open(join(run_dir, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters, indent=4))

    return run_dir


with open(
    join("flows/generate_train_data_2DTime_to_2D.md"),
    encoding="UTF-8",
) as f:
    description = f.read()


@task(cache_key_fn=task_input_hash, refresh_cache=True)
def list_images(
    input_dir: str,
    pattern: str,
    pixel_resolution_um: float,
    axes: str,
):
    pattern_re = re.compile(pattern)
    images: list[ImageSource] = []
    for entry in os.scandir(input_dir):
        if entry.is_file():
            if pattern_re.fullmatch(entry.name):
                images.append(
                    ImageSource.from_path(
                        entry.path,
                        metadata={
                            "axes": axes,
                            "unit": "micron",
                        },
                        resolution=[
                            1e4 / pixel_resolution_um,
                            1e4 / pixel_resolution_um,
                        ],
                    )
                )

    get_run_logger().info(f"Found {len(images)} images.")

    return images


@flow(
    name="N2V: Generate Train Data from 2D+Time",
    description=description,
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("prefect-n2v"),
)
def generate_train_data_2DTime_to_2D(
    user: User,
    run_name: str,
    input_data: InputData = InputData(),
    output_dir: str = "/tungstenfs/scratch",
    datagen_2d: DataGen2D = DataGen2D(),
) -> tuple[NumpyTarget, NumpyTarget]:
    run_dir = validate_parameters(
        user=user,
        run_name=run_name,
        input_data=input_data,
        output_dir=output_dir,
        datagen_2d=datagen_2d,
    )

    logger = get_run_logger()
    logger.info(f"Run logs are written to: {run_dir}")
    logger.info(f"N2V training data is save in: {output_dir}")

    img_files = list_images(
        input_dir=input_data.input_dir,
        pattern=input_data.pattern,
        pixel_resolution_um=input_data.xy_pixelsize_um,
        axes=input_data.axes,
    )

    x_train, x_val = extract_patches(
        img_files=img_files,
        num_patches_per_img=datagen_2d.num_patches_per_img,
        patch_shape=datagen_2d.patch_shape,
        output_dir=output_dir,
    )

    log_infrastructure(run_dir)

    return x_train, x_val

import json
import random
from copy import copy
from os.path import join

import numpy as np
import prefect
from cpr.image.ImageSource import ImageSource
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.utilities.utilities import task_input_hash
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from prefect import task
from prefect.artifacts import create_table_artifact

from flows.storage_keys import RESULT_STORAGE_KEY


def write_summary(path, n_patches, file_list):
    summary = (
        "# Summary\n"
        f"{n_patches} patches were extracted from the following image "
        f"files:\n"
        f"{json.dumps(file_list, indent=4)}"
    )

    with open(path, "w") as f:
        f.write(summary)


def move_axes_to_TZYXC(data, axes: str):
    source, destination = (), ()
    i = 0
    for c in "TZYXC":
        if c in axes:
            source += (axes.index(c),)
            destination += (i,)
            i += 1

    return np.moveaxis(data, source, destination)


@task(
    cache_key_fn=task_input_hash,
    result_storage_key=RESULT_STORAGE_KEY,
)
def extract_patches(
    img_files: list[ImageSource],
    num_patches_per_img: int,
    patch_shape: list[int],
    axes: str,
    output_dir: str,
):
    img_files_shuffled = copy(img_files)
    random.shuffle(img_files_shuffled)
    datagen = N2V_DataGenerator()
    split = int(min(max(len(img_files) * 0.1, 1), 500))

    images = []
    for img in img_files_shuffled:
        data = img.get_data()

        assert data.ndim <= 5, "Data can have at most 5 dimensions: TZYXC"

        data = move_axes_to_TZYXC(data, axes)

        if "C" in axes:
            assert (
                data.shape[axes.index("C")] == 1
            ), "Only single channel images are supported."

        if "T" in axes and "Z" in axes:
            data = np.concatenate(data, axis=0)

        if data.ndim == 2:
            data = data[np.newaxis, ..., np.newaxis]

        if data.ndim == 3:
            data = data[..., np.newaxis]

        images.append(data)

    x_val_data = datagen.generate_patches_from_list(
        images[:split],
        num_patches_per_img=num_patches_per_img,
        shape=patch_shape,
        augment=False,
    )
    x_train_data = datagen.generate_patches_from_list(
        images[split:],
        num_patches_per_img=num_patches_per_img,
        shape=patch_shape,
        augment=True,
    )

    val_output_file = join(output_dir, "x_val_2D.npy")
    x_val = NumpyTarget.from_path(val_output_file)
    x_val.set_data(x_val_data)
    create_table_artifact(
        key=f"{prefect.runtime.flow_run.name}-validation-images",
        table={
            "file_name": [img.name + img.ext for img in img_files_shuffled[:split]],
            "location": [img.location for img in img_files_shuffled[:split]],
        },
        description=f"From each 2D plane of each image file "
        f"{num_patches_per_img} patches of shape {patch_shape} "
        f"were extracted and saved in {val_output_file}.",
    )
    write_summary(
        path=join(output_dir, "x_val_2D.md"),
        n_patches=x_val_data.shape[0],
        file_list=[img.get_path() for img in img_files_shuffled[:split]],
    )

    train_output_file = join(output_dir, "x_train_2D.npy")
    x = NumpyTarget.from_path(train_output_file)
    x.set_data(x_train_data)
    create_table_artifact(
        key=f"{prefect.runtime.flow_run.name}-training-images",
        table={
            "file_name": [img.name + img.ext for img in img_files_shuffled[split:]],
            "location": [img.location for img in img_files_shuffled[split:]],
        },
        description=f"From each 2D plane of each image file "
        f"{num_patches_per_img} patches of shape {patch_shape} "
        f"were extracted and saved in {train_output_file}.",
    )
    write_summary(
        path=join(output_dir, "x_train_2D.md"),
        n_patches=x_train_data.shape[0],
        file_list=[img.get_path() for img in img_files_shuffled[split:]],
    )

    return x, x_val

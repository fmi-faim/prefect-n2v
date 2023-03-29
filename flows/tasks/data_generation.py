import json
import random
from copy import copy
from os.path import join

import numpy as np
from cpr.image.ImageSource import ImageSource
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.utilities.utilities import task_input_hash
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from prefect import task


def write_summary(path, n_patches, file_list):
    summary = (
        "# Summary\n"
        f"{n_patches} patches were extracted from the following image "
        f"files:\n"
        f"{json.dumps(file_list, indent=4)}"
    )

    with open(path, "w") as f:
        f.write(summary)


@task(cache_key_fn=task_input_hash)
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

        assert data.ndim <= 4, "Data can have at most 4 dimensions: TYXC"
        if "C" in axes:
            assert (
                data.shape[axes.index("C")] == 1
            ), "Only single channel images are supported."

        if "T" in axes:
            data = np.moveaxis(data, axes.index("T"), 0)

        if data.ndim == 2:
            data = data[np.newaxis, ..., np.newaxis]

        if data.ndim == 3:
            data = data[..., np.newaxis]

        images.append(data)

    x_val = NumpyTarget.from_path(join(output_dir, "x_val_2D.npy"))
    x_val.set_data(
        datagen.generate_patches_from_list(
            images[:split],
            num_patches_per_img=num_patches_per_img,
            shape=patch_shape,
            augment=False,
        )
    )
    write_summary(
        path=join(output_dir, "x_val_2D.md"),
        n_patches=x_val.get_data().shape[0],
        file_list=[img.get_path() for img in img_files_shuffled[:split]],
    )

    x = NumpyTarget.from_path(join(output_dir, "x_train_2D.npy"))
    x.set_data(
        datagen.generate_patches_from_list(
            images[split:],
            num_patches_per_img=num_patches_per_img,
            shape=patch_shape,
            augment=True,
        )
    )
    write_summary(
        path=join(output_dir, "x_train_2D.md"),
        n_patches=x.get_data().shape[0],
        file_list=[img.get_path() for img in img_files_shuffled[split:]],
    )

    return x, x_val

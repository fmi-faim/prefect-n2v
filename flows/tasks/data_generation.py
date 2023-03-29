from os.path import join

import numpy as np
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.utilities.utilities import task_input_hash
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from prefect import task


@task(cache_key_fn=task_input_hash)
def extract_patches(
    img_files: list[str],
    num_patches_per_img: int,
    patch_shape: list[int],
    output_dir: str,
):
    np.random.seed(42)
    idx = np.random.permutation(len(img_files))
    datagen = N2V_DataGenerator()
    split = int(min(max(len(img_files) * 0.1, 1), 500))

    x_val = NumpyTarget.from_path(join(output_dir, "x_val_2D.npy"))
    x_val.set_data(
        datagen.generate_patches_from_list(
            img_files[idx][:split],
            num_patches_per_img=num_patches_per_img,
            shape=patch_shape,
            augment=False,
        )
    )

    x = NumpyTarget.from_path(join(output_dir, "x_train_2D.npy"))
    x.set_data(
        datagen.generate_patches_from_list(
            img_files[idx][split:],
            num_patches_per_img=num_patches_per_img,
            shape=patch_shape,
            augment=True,
        )
    )

    return x, x_val

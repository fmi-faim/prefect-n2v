import os
import threading
from os.path import join
from threading import Semaphore
from typing import Any, Optional

import n2v.models
import numpy as np
import psutil
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from keras.backend import clear_session
from n2v.models import N2V
from prefect import get_run_logger, task
from prefect.context import TaskRunContext

from flows.storage_keys import RESULT_STORAGE_KEY
from flows.tasks.data_generation import move_axes_to_TZYXC


def exlude_semaphore_and_model_task_input_hash(
    context: "TaskRunContext", arguments: dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if (not isinstance(item, threading.Semaphore)) and (
            not isinstance(item, n2v.models.N2V)
        ):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


def move_axes_from_TZYXC(data, axes: str):
    source, destination = (), ()
    i = 0
    for c in "TZYXC":
        if c in axes:
            source += (i,)
            destination += (axes.index(c),)
            i += 1

    return np.moveaxis(data, source, destination)


@task(
    cache_key_fn=exlude_semaphore_and_model_task_input_hash,
    result_storage_key=RESULT_STORAGE_KEY,
)
def predict_2d(
    model: N2V,
    img: ImageSource,
    output_dir: str,
    gpu_semaphore: Semaphore,
):
    data = img.get_data()
    metadata = img.get_metadata()
    axes = metadata["axes"]
    if "C" not in axes:
        axes += "C"
        data = data[..., np.newaxis]

    dtype = data.dtype

    data = move_axes_to_TZYXC(data, axes)

    try:
        gpu_semaphore.acquire()
        if data.ndim == 5:
            pred = np.zeros_like(data)
            for t in range(data.shape[0]):
                for z in range(data.shape[1]):
                    pred[t, z] = np.clip(
                        model.predict(
                            img=data[t, z].astype(np.float32),
                            axes="YXC",
                        ),
                        np.iinfo(dtype).min,
                        np.iinfo(dtype).max,
                    ).astype(dtype)
        elif data.ndim == 4:
            pred = np.zeros_like(data)
            # Either z or t stack
            for s in range(data.shape[0]):
                pred[s] = np.clip(
                    model.predict(
                        img=data[s].astype(np.float32),
                        axes="YXC",
                    ),
                    np.iinfo(dtype).min,
                    np.iinfo(dtype).max,
                ).astype(dtype)
        elif data.ndim == 3:
            pred = np.clip(
                model.predict(
                    img=data.astype(np.float32),
                    axes="YXC",
                ),
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
            ).astype(dtype)
    except Exception as e:
        raise e
    finally:
        clear_session()
        gpu_semaphore.release()

    pred = move_axes_from_TZYXC(pred, axes)

    if "C" not in metadata["axes"]:
        pred = pred[..., 0]

    output: ImageTarget = ImageTarget.from_path(
        join(output_dir, img.get_name() + "_denoised" + img.ext)
    )
    output.set_metadata(img.get_metadata())
    output.set_resolution(tuple(img.get_resolution()))
    output.set_data(pred)

    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    get_run_logger().info(f"[End] Process memory usage: {mem_usage} GB")

    return output

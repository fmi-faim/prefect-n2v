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
    dtype = data.dtype

    try:
        gpu_semaphore.acquire()
        if "T" in axes:
            data = np.moveaxis(data, axes.index("T"), 0)
            pred = np.zeros_like(data)
            for i in range(len(data)):
                pred[i] = np.clip(
                    model.predict(
                        img=data[i].astype(np.float32),
                        axes="YX",
                    ),
                    np.iinfo(dtype).min,
                    np.iinfo(dtype).max,
                ).astype(dtype)
        else:
            pred = model.predict(
                img=data.astype(np.float32),
                axes=axes,
            )
    except Exception as e:
        raise e
    finally:
        clear_session()
        gpu_semaphore.release()

    output: ImageTarget = ImageTarget.from_path(
        join(output_dir, img.get_name() + "_denoised" + img.ext)
    )
    output.set_metadata(img.get_metadata())
    output.set_resolution(tuple(img.get_resolution()))
    output.set_data(np.moveaxis(pred, 0, axes.index("T")))

    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    get_run_logger().info(f"[End] Process memory usage: {mem_usage} GB")

    return output

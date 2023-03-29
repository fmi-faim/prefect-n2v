from pydantic import BaseModel


class InputData(BaseModel):
    input_dir: str = "/tungstenfs/scratch"
    pattern: str = ".*.tif"
    axes: str = "YX"
    xy_pixelsize_um: float = 0.134


class DataGen2D(BaseModel):
    patch_shape: list[int] = [96, 96]
    num_patches_per_img: int = 8


class TrainData(BaseModel):
    train_data: str = "/tungstenfs/scratch"
    val_data: str = "/tungstenfs/scratch"


class N2VModel(BaseModel):
    output_dir: str = "/tungstenfs/scratch"
    model_name: str = "n2v2-model"
    epochs: int = 200
    batch_size: int = 128
    unet_depth: int = 2


class WandB(BaseModel):
    host: str = "https://wandb.fmi.ch"
    secret_key: str = "wandb-user"
    project: str = "denoising"
    entity: str = "faim"

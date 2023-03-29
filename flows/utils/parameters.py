from pydantic import BaseModel


class InputData(BaseModel):
    input_dir: str = "/tungstenfs/scratch"
    pattern: str = ".*.tif"
    axes: str = "YX"
    xy_pixelsize_um: float = 0.134


class DataGen2D(BaseModel):
    patch_shape: list[int] = [96, 96]
    num_patches_per_img: int = 8

# N2V: Generate Train Data from 2D+Time
Creates training data to train a [N2V](https://github.com/juglab/n2v) denoising network.

## Input Format
The provided tiff-files must be 2D.

## Flow Parameters
* `user`:
    * `name`: Name of the user.
    * `group`: Group name of the user.
* `run_name`: Name of processing run.
* `input_data`:
    * `input_dir`: Input directory containing the 2D+Channel tiff files.
    * `pattern`: A pattern to filter the tiff files.
    * `axes`: String indicating the axes order of the tiff files.
    * `xy_pixelsize_um`: The pixel-size in micrometers.
* `output_dir`: Path to the output directory.
* `datagen_2d`: 
    * `patch_shape`: Size of the extracted training patches.
    * `num_patches_per_img`: Number of patches extracted from each image.

## Output Format
Two files `x_train_2D.npy` and `x_val_2D.npy` are created and saved in `output_dir`.

## Citation
If you use this flow please cite the [N2V](https://arxiv.org/abs/1811.10980) and [N2V2](https://arxiv.org/abs/2211.08512) publications:
```text
@inproceedings{krull2019noise2void,
  title={Noise2void-learning denoising from single noisy images},
  author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2129--2137},
  year={2019}
}
@inproceedings{hock2023n2v2,
  title={N2V2-Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a Tweaked Network Architecture},
  author={H{\"o}ck, Eva and Buchholz, Tim-Oliver and Brachmann, Anselm and Jug, Florian and Freytag, Alexander},
  booktitle={Computer Vision--ECCV 2022 Workshops: Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={503--518},
  year={2023},
  organization={Springer}
}
```
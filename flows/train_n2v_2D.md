# N2V: Train 2D model
Trains a 2D [N2V2](https://github.com/juglab/n2v) denoising model.

## Input Format
Expects training data generated with the 'N2V: Generate Train Data from 2D+Time' flow.

## Flow Parameters
* `user`:
    * `name`: Name of the user.
    * `group`: Group name of the user.
* `run_name`: Name of processing run.
* `train_data`:
    * `train_data`: Path to the `x_train_2D-...npy` file.
    * `val_data`: Path to the `x_val_2D-...npy` file.
* `n2v_model`:
    * `output_dir`: Directory where the model will be saved.
    * `model_name`: Name of the denoising model.
    * `epochs`: Number of epochs for which the model is trained.
    * `batch_size`: Training batch size.
    * `unet_depth`: Number of down-sampling layers.

## Output Format
Saves a trained model in `n2v_model.output_dir`.

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

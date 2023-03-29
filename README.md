# prefect-n2v
A collection of prefect flows to run [noise2void](https://github.com/juglab/n2v).

## Installation
Create a new conda environment and install the requirements.
```shell
conda env create -f environment.yaml
```

Or follow the official install instructions from [n2v](https://github.com/juglab/n2v#installation).


## WandB
We use [Weights & Biases]() to log network trainings. W&B uses local storage to save logs, artifacts and configs. It might be necessary to change [these locations](https://docs.wandb.ai/guides/artifacts/storage).

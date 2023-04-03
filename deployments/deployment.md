# Build
```shell
prefect deployment build flows/generate_train_data_2DTime_to_2D.py:generate_train_data_2DTime_to_2D -n "default" -q slurm -sb github/prefect-n2v --skip-upload -o deployments/generate_train_data_2DTime_to_2D.yaml  -ib process/prefect-n2v-cpu -t denoising -t 2D

prefect deployment build flows/train_n2v_2D.py:train_n2v_2d -n "default" -q slurm -sb github/prefect-n2v --skip-upload -o deployments/train_n2v_2d.yaml  -ib process/prefect-n2v-gpu -t denoising -t 2D

prefect deployment build flows/predict_n2v_2D.py:predict_n2v_2d -n "default" -q slurm -sb github/prefect-n2v --skip-upload -o deployments/predict_n2v_2d.yaml  -ib process/prefect-n2v-gpu -t denoising -t 2D
```

# Apply
```shell
prefect deployment apply deployments/*.yaml
```

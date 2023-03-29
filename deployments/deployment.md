# Build
```shell
prefect deployment build flows/generate_train_data_2DTime_to_2D.py:generate_train_data_2DTime_to_2D -n "default" -q slurm -sb github/prefect-n2v --skip-upload -o deployments/generate_train_data_2DTime_to_2D.yaml  -ib process/prefect-n2v-cpu -t denoising -t 2D
```

# Apply
```shell
prefect deployment apply deployments/*.yaml
```

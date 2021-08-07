# <div align="center">Audio Classification, Tagging in PyTorch</div>

## <div align="center">Model Zoo</div>

Model | ESC50 Acc <br><sup>(%) | Params <br><sup>(M)  | GFLOPs | Weights
--- | --- | --- | --- | --- 
ResNet50 | - | - | - | N/A

<details>
  <summary>Table Notes <small>(click to expand)</small></summary>

</details>


## <div align="center">Usage</div>

<details>
  <summary>Dataset Preparation <small>(click to expand)</small></summary>

</details>

<details>
  <summary>Configuration <small>(click to expand)</small></summary>

Create a configuration file in `configs`. Sample configuration for ImageNet dataset can be found [here](configs/defaults.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<details>
  <summary>Training <small>(click to expand)</small></summary>

Train with 1 GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

Train with 2 GPUs:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details>
  <summary>Training with KD <small>(click to expand)</small></summary>

Change `ENABLE` field in `KD` of the configuration file to `True` and also change the additional parameters. The weights file for the teacher model must be supplied via `PRETRAINED` field.

The training command is the same as in above.

</details>


<details>
  <summary>Evaluation <small>(click to expand)</small></summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details>
  <summary>Inference <small>(click to expand)</small></summary>

Make sure to set `MODEL_PATH` of the configuration file to model's weights.

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>

<details>
  <summary>Optimization <small>(click to expand)</small></summary>

For optimizing these models for deployment, see [torch_optimize](https://github.com/sithu31296/torch_optimize).

</details>

<details>
  <summary>References <small>(click to expand)</small></summary>


</details>
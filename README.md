# <div align="center">Audio Tagging & Sound Event Detection in PyTorch</div>

## <div align="center">Model Zoo</div>

[cnn14]: https://drive.google.com/file/d/1GhDXnyj9KgDMyOOoMuSBn8pb1iELlEp7/view?usp=sharing
[cnn1416k]: https://drive.google.com/file/d/1BGAfVH_6xt06YZUDPqRLNtyj7KoyoEaF/view?usp=sharing
[cnn14max]: https://drive.google.com/file/d/1K0XKf6JbFIgCoo70WvdunQoWWMMmrqDl/view?usp=sharing

<details open>
  <summary>Pretrained Models</small></summary>

Model | Task | AudioSet Accuracy <br><sup>(%) | Sample Rate <br><sup>(kHz) | Window Length | Num Mels | Fmax | Weights
--- | --- | --- | --- | --- | --- | --- | --- 
CNN14 | Tagging | 43.1 | 32 | 1024 | 64 | 14k | [download][cnn14]
CNN14_16k | Tagging | 43.8 | 16 | 512 | 64 | 8k | [download][cnn1416k]
||
CNN14_DecisionLevelMax | SED | 38.5 | 32 | 1024 | 64 | 14k | [download][cnn14max]

</details>

<details open>
  <summary>Fine-tuned Models</summary>

Model | Task | Dataset | Accuracy<br><sup>(%)  | Sample Rate <br><sup>(kHz) | Window Length | Num Mels | Fmax | Weights
--- | --- | --- | --- | --- | --- | --- | --- | --- 
CNN14 | Tagging | ESC50 | ? | 32 | 1024 | 64 | 14k | -

</details>

## <div align="center">Datasets</div>

[esc50]: https://github.com/karolpiczak/ESC-50
[fsdkaggle]: https://zenodo.org/record/2552860
[audioset]: https://research.google.com/audioset/
[urbansound8k]: https://urbansounddataset.weebly.com/urbansound8k.html
[speechcommands]: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

Dataset | Type | Classes | Train | Test | Samples<br>/class | Audio Length | Audio Spec | Size
--- | --- | --- | --- | --- | --- | --- | --- | --- 
[ESC-50][esc50] | Environmental | 50 | 2,000 | 5 folds | 40 | 5s | 44.1kHz, mono | 600MB
[FSDKaggle1028][fsdkaggle] | - | 41 | 9,473 | 1,600 | 94~300 | 300ms~30s | 44.1kHz, mono | 4.6GB
[UrbanSound8k][urbansound8k] | Urban | 10 | 8,732 | 10 folds | - | <=4s | - | 5.6GB
[SpeechCommands][speechcommands] | Words | 30 | 65,000 | - | - | 1s | - | 1.4GB

<details>
  <summary>Dataset Structure (click to expand)</small></summary>

```
datasets
|__ ESC50
    |__ audio

|__ FSDKaggle2018
    |__ audio_train
    |__ audio_test
    |__ FSDKaggle2018.meta
        |__ train_post_competition.csv
        |__ test_post_competition_scoring_clips.csv
```

</details>


## <div align="center">Usage</div>

<details>
  <summary>Configuration <small>(click to expand)</small></summary>

Create a configuration file in `configs`. Sample configuration for ImageNet dataset can be found [here](configs/tagging.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

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
  <summary>Evaluation <small>(click to expand)</small></summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details open>
  <summary>Inference</summary>

* Set `MODEL_PATH` of the configuration file to path of the model's weights.
* Set the testing audio file path in `TEST` >> `FILE`.

```bash
## audio tagging inference
$ python tools/tagging_infer.py --cfg configs/TAGGING_CONFIG_FILE.yaml

## sound event detection inference
$ python tools/sed_infer.py --cfg configs/SED_CONFIG_FILE.yaml
```

</details>

<details>
  <summary>Optimization <small>(click to expand)</small></summary>

For optimizing these models for deployment, see [torch_optimize](https://github.com/sithu31296/torch_optimize).

</details>

<details>
  <summary>References <small>(click to expand)</small></summary>


</details>
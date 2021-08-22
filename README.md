# <div align="center">Audio Classification, Tagging & Sound Event Detection in PyTorch</div>

Progress:

- [x] Fine-tune on audio classification
- [ ] Fine-tune on audio tagging
- [ ] Fine-tune on sound event detection
- [x] Add tagging metrics
- [ ] Add Tutorial
- [x] Add Augmentation Notebook
- [ ] Add more schedulers
- [ ] Add FSDKaggle2019 dataset
- [ ] Add MTT dataset
- [ ] Add DESED
- [ ] Test in real-time


## <div align="center">Model Zoo</div>

[cnn14]: https://drive.google.com/file/d/1GhDXnyj9KgDMyOOoMuSBn8pb1iELlEp7/view?usp=sharing
[cnn1416k]: https://drive.google.com/file/d/1BGAfVH_6xt06YZUDPqRLNtyj7KoyoEaF/view?usp=sharing
[cnn14max]: https://drive.google.com/file/d/1K0XKf6JbFIgCoo70WvdunQoWWMMmrqDl/view?usp=sharing

<details open>
  <summary><strong>AudioSet Pretrained Models</strong></summary>

Model | Task | mAP <br><sup>(%) | Sample Rate <br><sup>(kHz) | Window Length | Num Mels | Fmax | Weights
--- | --- | --- | --- | --- | --- | --- | --- 
CNN14 | Tagging | 43.1 | 32 | 1024 | 64 | 14k | [download][cnn14]
CNN14_16k | Tagging | 43.8 | 16 | 512 | 64 | 8k | [download][cnn1416k]
||
CNN14_DecisionLevelMax | SED | 38.5 | 32 | 1024 | 64 | 14k | [download][cnn14max]

</details>

> Note: These models will be used as a pretrained model in the fine-tuning tasks below. Check out [audioset-tagging-cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn), if you want to train on AudioSet dataset.  

[esc50cnn14]: https://drive.google.com/file/d/1itN-WyEL6Wp_jVBlld6vLaj47UWL2JaP/view?usp=sharing
[fsd2018]: https://drive.google.com/file/d/1KzKd4icIV2xF7BdW9EZpU9BAZyfCatrD/view?usp=sharing
[scv1]: https://drive.google.com/file/d/1Mc4UxHOEvaeJXKcuP4RiTggqZZ0CCmOB/view?usp=sharing

<details open>
  <summary><strong>Fine-tuned Classification Models</strong></summary>

Model | Dataset | Accuracy<br><sup>(%) | Sample Rate <br><sup>(kHz) | Weights
--- | --- | --- | --- | ---  
CNN14 | ESC50 (Fold-5)| 95.75 | 32 | [download][esc50cnn14]
CNN14 | FSDKaggle2018 (test) | 93.56 | 32 | [download][fsd2018]
CNN14 | SpeechCommandsv1 (val/test) | 96.60/96.77 | 32 | [download][scv1]

</details>

<details>
  <summary><strong>Fine-tuned Tagging Models</strong></summary>

Model | Dataset | mAP(%)  | AUC | d-prime | Sample Rate <br><sup>(kHz) | Config | Weights
--- | --- | --- | --- | --- | --- | --- | ---
CNN14 | FSDKaggle2019 | - | - | - | 32 | - | -

</details>

<details>
  <summary><strong>Fine-tuned SED Models</strong></summary>

Model | Dataset | F1 | Sample Rate <br><sup>(kHz) | Config | Weights
--- | --- | --- | --- | --- | ---
CNN14_DecisionLevelMax | DESED | - | 32 | - | -

</details>

---

## <div align="center">Supported Datasets</div>

[esc50]: https://github.com/karolpiczak/ESC-50
[fsdkaggle2018]: https://zenodo.org/record/2552860
[fsdkaggle2019]: https://zenodo.org/record/3612637
[audioset]: https://research.google.com/audioset/
[urbansound8k]: https://urbansounddataset.weebly.com/urbansound8k.html
[speechcommandsv1]: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
[speechcommandsv2]: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
[mtt]: https://github.com/keunwoochoi/magnatagatune-list
[desed]: https://project.inria.fr/desed/

Dataset | Task | Classes | Train | Val | Test | Audio Length | Audio Spec | Size
--- | --- | --- | --- | --- | --- | --- | --- | --- 
[ESC-50][esc50] | Classification | 50 | 2,000 | 5 folds | - | 5s | 44.1kHz, mono | 600MB
[UrbanSound8k][urbansound8k] | Classification | 10 | 8,732 | 10 folds | - | <=4s | Vary | 5.6GB
[FSDKaggle2018][fsdkaggle2018] | Classification | 41 | 9,473 | - | 1,600 | 300ms~30s | 44.1kHz, mono | 4.6GB
[SpeechCommandsv1][speechcommandsv1] | Classification | 30 | 51,088 | 6,798 | 6,835 | <=1s | 16kHz, mono | 1.4GB
[SpeechCommandsv2][speechcommandsv2] | Classification | 35 | 84,843 | 9,981 | 11,005 | <=1s | 16kHz, mono | 2.3GB
||
[FSDKaggle2019][fsdkaggle2019]* | Tagging | 80 | 4,970+19,815 | - | 4,481 | 300ms~30s | 44.1kHz, mono | 24GB
[MTT][mtt]* | Tagging | 50 | 19,000 | - | - | - | - | 3GB
||
[DESED][desed]* | SED | 10 | - | - | - | 10 | - | -

> Notes: `*` datasets are not available yet. Classification dataset are treated as multi-class/single-label classification and tagging and sed datasets are treated as multi-label classification.

<details>
  <summary><strong>Dataset Structure</strong> (click to expand)</summary>

Download the dataset and prepare it into the following structure.

```
datasets
|__ ESC50
    |__ audio

|__ Urbansound8k
    |__ audio

|__ FSDKaggle2018
    |__ audio_train
    |__ audio_test
    |__ FSDKaggle2018.meta
        |__ train_post_competition.csv
        |__ test_post_competition_scoring_clips.csv

|__ SpeechCommandsv1/v2
    |__ bed
    |__ bird
    |__ ...
    |__ testing_list.txt
    |__ validation_list.txt

```

</details>

<details>
  <summary><strong>Augmentations</strong> (click to expand)</summary>

Currently, the following augmentations are supported. More will be added in the future. You can test the effects of augmentations with this [notebook](./datasets/aug_test.ipynb)

WaveForm Augmentations:

- [x] MixUp 
- [x] Background Noise
- [x] Gaussian Noise
- [x] Fade In/Out 
- [x] Volume
- [ ] CutMix

Spectrogram Augmentations:

- [x] Time Masking
- [x] Frequency Masking
- [x] Filter Augmentation

</details>

---

## <div align="center">Usage</div>

<details>
  <summary><strong>Requirements</strong> (click to expand)</summary>

* python >= 3.6
* torch >= 1.8.1
* torchaudio >= 0.8.1

Other requirements can be installed with `pip install -r requirements.txt`.

</details>

<br>
<details>
  <summary><strong>Configuration</strong> (click to expand)</summary>

* Create a configuration file in [configs](./configs/). Sample configuration for ESC50 dataset can be found [here](configs/esc50.yaml). 
* Copy the contents of this and then edit the fields you think if it is needed. 
* This configuration file is needed for all of training, evaluation and prediction scripts.

</details>
<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

To train with a single GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

To train with multiple gpus, set `DDP` field in config file to `true` and run as follows:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>

<br>
<details>
  <summary><strong>Evaluation</strong> (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE.yaml
```

</details>

<br>
<details open>
  <summary><strong>Audio Classification/Tagging Inference</strong></summary>

* Set `MODEL_PATH` of the configuration file to your model's trained weights.
* Change the dataset name in `DATASET` >> `NAME` as your trained model's dataset.
* Set the testing audio file path in `TEST` >> `FILE`.
* Run the following command.

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE.yaml

## for example
$ python tools/infer.py --cfg configs/audioset.yaml
```
You will get an output similar to this:

```bash
Class                     Confidence
----------------------  ------------
Speech                     0.897762
Telephone bell ringing     0.752206
Telephone                  0.219329
Inside, small room         0.20761
Music                      0.0770325
```

</details>

<br>
<details open>
  <summary><strong>Sound Event Detection Inference</strong></summary>

* Set `MODEL_PATH` of the configuration file to your model's trained weights.
* Change the dataset name in `DATASET` >> `NAME` as your trained model's dataset.
* Set the testing audio file path in `TEST` >> `FILE`.
* Run the following command.

```bash
$ python tools/sed_infer.py --cfg configs/CONFIG_FILE.yaml

## for example
$ python tools/sed_infer.py --cfg configs/audioset_sed.yaml
```

You will get an output similar to this:

```bash
Class                     Start    End
----------------------  -------  -----
Speech                      2.2    7
Telephone bell ringing      0      2.5
```

The following plot will also be shown, if you set `PLOT` to `true`:

![sed_result](./assests/sed_result.png)

</details>

<br>
<details>
  <summary><strong>References</strong> (click to expand)</summary>

* https://github.com/qiuqiangkong/audioset_tagging_cnn
* https://github.com/YuanGongND/ast
* https://github.com/frednam93/FilterAugSED
* https://github.com/lRomul/argus-freesound

</details>

<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

```
@misc{kong2020panns,
      title={PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition}, 
      author={Qiuqiang Kong and Yin Cao and Turab Iqbal and Yuxuan Wang and Wenwu Wang and Mark D. Plumbley},
      year={2020},
      eprint={1912.10211},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}

@misc{gong2021ast,
      title={AST: Audio Spectrogram Transformer}, 
      author={Yuan Gong and Yu-An Chung and James Glass},
      year={2021},
      eprint={2104.01778},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}

@misc{nam2021heavily,
      title={Heavily Augmented Sound Event Detection utilizing Weak Predictions}, 
      author={Hyeonuk Nam and Byeong-Yun Ko and Gyeong-Tae Lee and Seong-Hu Kim and Won-Ho Jung and Sang-Min Choi and Yong-Hwa Park},
      year={2021},
      eprint={2107.03649},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

</details>
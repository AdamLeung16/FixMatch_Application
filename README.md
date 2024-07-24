# FixMatch_Application

A semi-supervised image classification task using FixMatch, on CIFAR-10 dataset.

### 1. Introduction

In this project, I use 40, 250 and 4000 labeled data, together with other unlabeled data in CIFAR-10 dataset. I also make comparisons between my FixMatch and the one made by TorchSSL.

### 2. my FixMatch

Using WideResNet as backbone.

### 3. TorchSSL FixMatch

You can download the source code from the following link. [https://github.com/StephenStorm/TorchSSL](https://github.com/StephenStorm/TorchSSL "https://github.com/StephenStorm/TorchSSL")

* cifar10-40
  ```shell
  python fixmatch.py --c config/fixmatch/fixmatch_cifar10_40_0.yaml
  ```
* cifar10-250
  ```shell
  python fixmatch.py --c config/fixmatch/fixmatch_cifar10_250_0.yaml
  ```
* cifar10-4000
  ```shell
  python fixmatch.py --c config/fixmatch/fixmatch_cifar10_4000_0.yaml
  ```

### 4. Results

Accuracy comparisons between my FixMatch and TorchSSL FixMatch, at 40000 iteration.

| labeled num | 40                     | 250                    | 4000                    |
| ----------- | ---------------------- | ---------------------- | ----------------------- |
| my FixMatch | 44%                    | 67%                    | 86%                     |
| TorchSSL    | 58.7% (Finally 91.21%) | 90.4% (Finally 93.74%) | 91.94% (Finally 94.49%) |

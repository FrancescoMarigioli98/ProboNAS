# ProboNAS
ProboNas is a block-based evoltionary Neural Architecture Search algorithm.
This repository contains the code for our NAS for Tiny Visual Wake Words, realized for the Machine Learning and Deep Learning class (A.Y. 2022-2023).

We specifically tackled the task of human detection using the VisualWakeWords dataset, which is part of the COCO-Dataset and labels the presence or absence of a "Person". In this paper, we propose ProboNAS, a block-based evolutionary NAS algorithm that combines training-free metrics, customized block search space, and an efficient evolutionary algorithm. It achieves remarkable results by quickly finding architectures that perform excellently while respecting the computational and memory limits of modern devices, without the need for network training. Our experiments demonstrate that ProboNAS is a rapid, efficient, and successful approach for automated model design, representing a valuable solution for the task of human detection on the evaluated dataset.


### Dependencies
* Python 3 (tested on python 3.6)
* PyTorch
  * with GPU and CUDA enabled installation (though the code is not runnable on CPU)
 
### Execution Guide
You can Previously download and compress the dataset using this script: [![Dataset download](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zPJwTzZyh9xpxfFbhV1dmHH-FYe0ITa-#scrollTo=qkIbAzgB02De)

Next, an example to run our work is available on colab at the following link: [![Dataset download](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13iaHOeUwYgUas3jofnPpXukBllm6fypu#scrollTo=W3tARQvUGkbb)

## Pipeline Description
Our search space is block-based. Thanks to the evaluation metrics, the ProboNAS algorithm can evaluate the networks and proceed with the searching for the best candidate. The feature map going throught the architecture extracted by ProboNAS has this structure:

<div align="center">
  <img src="https://github.com/FrancescoMarigioli98/ProboNAS/blob/main/imgs/Network.png" alt="Alt text" title="Network" width="700" height="300">
</div>

The inner blocks can have two possible configuration:
1. *MobileNet Block*.

<div align="center">
  <img src="https://github.com/FrancescoMarigioli98/ProboNAS/blob/main/imgs/Mobilenet.png" alt="Alt text" title="Preprocessing pipeline" width="700" height="300">
</div>




2. *ConvNeXt Block*.

<div align="center">
  <img src="https://github.com/FrancescoMarigioli98/ProboNAS/blob/main/imgs/Convnext.png" alt="Alt text" title="Preprocessing pipeline" width="700" height="300">
</div>

<p>

We started with a random population and selected parents for reproduction based on a Tournament selection algorithm. Mutations and crossovers were applied to create diverse individuals. Constraints on FLOPs and parameters were set, and feasible individuals were added to the population. The algorithm identified the top architectures based on evaluation metrics. The best architecture's code was provided at the end, evaluated using training-free metrics.

Please, for further information, and to see our results, refere to the relative [`paper`](https://github.com/FrancescoMarigioli98/ProboNAS/blob/main/ProboNAS.pdf).

----

Authors: [`Dario Avalle`](https://github.com/davalle99) [`Michela Saraceno`](https://github.com/michelasaraceno) [`Francesco Marigioli`](https://github.com/FrancescoMarigioli98) Politecnico di Torino. 

# RetroDFM-R: Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning

<div align="center" style="line-height: 1;">
 
[![Paper](https://img.shields.io/badge/PAPER-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.17448)
[![GitHub](https://img.shields.io/badge/GITHUB-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OpenDFM/RetroDFM-R) 
[![Model](https://img.shields.io/badge/MODEL-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/OpenDFM/RetroDFM-R-8B)
[![License](https://img.shields.io/badge/LICENSE-green?style=for-the-badge&logo=open-source-initiative&logoColor=white)](https://github.com/OpenDFM/RetroDFM-R/blob/main/LICENSE)

</div>

This repository contains the code and resources for "Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning," which introduces RetroDFM-R.

## 🔥News

- [2025-11-22] We released the parameters of RetroDFM-R-8B on [🤗 Hugging Face](https://huggingface.co/OpenDFM/RetroDFM-R-8B).
- [2025-07-23] The paper of RetroDFM-R is available on arXiv: [Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2507.17448).


## 🛠️ Setup

### Training Environment

We recommend using the provided Docker image for training. Follow the installation instructions for OpenRLHF to set up your environment.

```bash
# Follow OpenRLHF installation guide
# https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation
# We recommend using the Docker image for training.
```

Once inside the Docker container, install `rdkit`:

```bash
pip install rdkit
```

### Inference Environment

To set up the environment for inference, follow these steps:

```bash
conda create -n retrodfmR python=3.10
conda activate retrodfmR
pip install -r requirements.txt
```

You can specify the CUDA version if needed (e.g., for CUDA 12.8):

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128 # Replace cu128 with your CUDA version
```

## Data Preparation

All data used in this work are sourced from publicly accessible datasets:

  * **SMILES/IUPAC Name Conversion Data**: Paired SMILES and IUPAC names are obtained from [PubChem](https://pubchem.ncbi.nlm.nih.gov/), used to construct the name conversion data.
  * **Retrosynthesis Data**:
      * **USPTO-50K**: Accessed via the GLN repository: https://github.com/Hanjun-Dai/GLN (specifically, the `schneider50k` dataset).
      * **USPTO-FULL**: Also obtained from the GLN repository: https://github.com/Hanjun-Dai/GLN (specifically, `uspto_multi` dataset).

We provide the processed test data on Hugging Face:
[https://huggingface.co/datasets/OpenDFM/retrodfm-R-inference](https://huggingface.co/datasets/OpenDFM/retrodfm-R-inference)

## 🏋️‍♂️ Training

Ensure your Docker container is successfully launched before initiating training.

Navigate to the `train` directory:

```bash
cd train
```

Then, execute the following training script:

  * **For Continual Pretraining:**
    ```bash
    bash examples/scripts/train_continual_pretrain.sh
    ```
  * **For Cold-Start Distillation:**
    ```bash
    bash examples/scripts/train_cold_start_distill.sh
    ```
  * **For Reinforcement Learning:**
      * On USPTO-50K:
        ```bash
        bash examples/scripts/train_dapo_retrodfm_R_50k.sh
        ```
      * On USPTO-FULL:
        ```bash
        bash examples/scripts/train_dapo_retrodfm_R_full.sh
        ```

## 🚀 Inference

After downloading the processed test data (as mentioned in Data Preparation), you can run the inference script. The following command will perform inference using beam search and test augmentation:

```bash
conda activate retrodfmR
cd inference && bash eval.sh
```

## 📖 Citation

Please cite our paper if you find our work useful:

```bibtex
@misc{zhang2025retrodfmr,
  title={Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning},
  author={Zhang, Situo and Li, Hanqi and Chen, Lu and Zhao, Zihan and Lin, Xuanze and Zhu, Zichen and Chen, Bo and Chen, Xin and Yu, Kai},
  year={2025},
  eprint={2507.17448},
  archivePrefix={arXiv},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2507.17448}, 
}
```
# Rewrite of CTAB_GAN+ 
This is a rewrite of the CTAB-GAN+ found in https://github.com/Team-TUD/CTAB-GAN-Plus, which should be credited with their work on a solid tabular data GAN. 

## Ongoing project
This is part of an ongoing project trying speed up and implement new features to this framework. The readme file from the original github follows below.

# NEWS! - 19/11/2023
Our new paper [TabuLa: Harnessing Language Models for Tabular Data Synthesis](https://arxiv.org/abs/2310.12746) is on arxiv now! The code is published 
[here](https://github.com/zhao-zilong/Tabula). Tabula improves tabular data synthesis by leveraging language model structures without the burden of pre-trained model weights. It offers a faster training process by preprocessing tabular data to shorten token sequence, which sharply reducing training time while consistently delivering higher-quality synthetic data. Its training time is longer than CTAB-GAN+, but the synthetic data fidelity is amazing! **It also works for high-dimentional categorical columns!**

# CTAB-GAN+
This is the official git paper [CTAB-GAN+: Enhancing Tabular Data Synthesis](https://arxiv.org/abs/2204.00401). Current code is **WITHOUT** differential privacy part. The code with differential privacy is in this [github](https://github.com/Team-TUD/CTAB-GAN-Plus-DP). 
If you have any question, please contact `z.zhao-8@tudelft.nl` for more information.


## Prerequisite

The required package version
```
numpy==1.21.0
torch==1.9.1
pandas==1.2.4
sklearn==0.24.1
dython==0.6.4.post1
scipy==1.4.1
```
The sklean package in newer version has updated its function for `sklearn.mixture.BayesianGaussianMixture`. Therefore, user should use this proposed sklearn version to successfully run the code!

## Example
`Experiment_Script_Adult.ipynb`  `Experiment_Script_king.ipynb` are two example notebooks for training CTAB-GAN+ with Adult (classification) and king (regression) datasets. The datasets are alread under `Real_Datasets` folder.
The evaluation code is also provided.

## Problem type

You can either indicate your dataset problem type as Classification, Regression. If there is no problem type, you can leave the problem type as None as follows:
```
problem_type= {None: None}
```

## For large dataset

If your dataset has large number of column, you may encounter the problem that our currnet code cannot encode all of your data since CTAB-GAN+ will wrap the encoded data into an image-like format. What you can do is changing the line 378 and 385 in `model/synthesizer/ctabgan_synthesizer.py`. The number in the `slide` list
```
sides = [4, 8, 16, 24, 32]
```
is the side size of image. You can enlarge the list to [4, 8, 16, 24, 32, 64] or [4, 8, 16, 24, 32, 64, 128] for accepting larger dataset.

## Bibtex

To cite this paper, you could use this bibtex

```
@article{zhao2023ctab,
  title={Ctab-gan+: Enhancing tabular data synthesis},
  author={Zhao, Zilong and Kunar, Aditya and Birke, Robert and Van der Scheer, Hiek and Chen, Lydia Y},
  journal={Frontiers in big Data},
  volume={6},
  year={2023},
  publisher={Frontiers Media SA}
}
```

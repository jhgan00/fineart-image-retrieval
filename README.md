# Fine Art Image Retrieval

This repository is based on the pytorch implementation of [Masked Autoencoders Are Scalable Vision Learners]()(He et al., 2021).

![img.png](assets/img.png)

## Architecture

We fine-tuned MAE model at style classification, genre classification and knn triplet learning task simultaneously.

![img_1.png](assets/img_1.png)

## K-NN Triplet Loss


## Experiments

- [WikiArt](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)
  - Notice) We used only images that had both style and genre labels.
- [MulititasPainting100k](http://www.ivl.disco.unimib.it/activities/paintings/)

![img_2.png](assets/img_2.png)
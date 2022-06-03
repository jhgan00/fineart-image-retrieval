# Fine Art Image Retrieval

This repository is based on the pytorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://github.com/facebookresearch/mae)(He et al., 2021).

![img.png](assets/img.png)

## Architecture

We fine-tuned MAE model at style classification, genre classification and knn triplet learning task simultaneously.

![img_1.png](assets/img_1.png)

## K-NN Triplet Loss

For each data point $x^{(b)}$ in a minibatch and the nearest neighbors $x_1^{(b)}, x_2^{(b)}, \cdots, x_K^{(b)}$, we define the pairwise relevance measure $r_i^{(b)}$ as:

![image](https://user-images.githubusercontent.com/47095378/171817828-3ee952d6-77f8-4c34-bba2-7b25d524bda6.png)

and the knn triplet loss as:

![image](https://user-images.githubusercontent.com/47095378/171816527-90668840-5b73-461e-8b8b-c367dfeccae3.png)



## Experiments

- [WikiArt](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)
  - Notice) We used only images that had both style and genre labels.
- [MulititasPainting100k](http://www.ivl.disco.unimib.it/activities/paintings/)

![img_2.png](assets/img_2.png)

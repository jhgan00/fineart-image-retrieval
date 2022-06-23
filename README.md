# Fine Art Image Retrieval

This repository is based on the pytorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://github.com/facebookresearch/mae)(He et al., 2021).

![img.png](assets/img.png)

## Architecture

The system trains MAE model with style classification, genre classification and triplet learning task simultaneously.

<p align="center">
  <img src="assets/img_1.png" style="height:450px; align:center" >
</p>



## KNN Triplet Loss

For each data point $x^{(a)}$ in a minibatch and the $k$-nearest neighbors $x_1^{(a)}, x_2^{(a)}, \cdots, x_k^{(a)}$, the knn triplet loss is defined as:

![image](https://user-images.githubusercontent.com/47095378/175226382-c2ca7762-fd74-4525-a469-5b883af84bbd.png)

where the relevance measure $r_i^{(a)}$ is defined as:

$$r_i^{(a)}= \begin{cases}
1 & \text{ if } s^{(a)} = s_i^{(a)} \text{ and } g^{(a)} = g_i^{(a)} \\
0 & \text{ if } s^{(a)} \neq s_i^{(a)} \text{ and } g^{(a)} \neq g_i^{(a)} \\
0.5 &  \text{ otherwise}
\end{cases}
$$

## Experiments

- [WikiArt](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)
  - Notice) We used only images that had both style and genre labels.
- [MulititasPainting100k](http://www.ivl.disco.unimib.it/activities/paintings/)

<table style="text-align:center" >
 <tr>
  <td rowspan="3">Loss Function</td>
  <td colspan="6">Wikiart paintings</td>
  <td colspan="6">MultitaskPainting100k</td>
 </tr>
 <tr>
  <td colspan="3">Style</td>
  <td colspan="3">Genre</td>
  <td colspan="3">Style</td>
  <td colspan="3">Genre</td>
 </tr>
 <tr>
  <td>P@1</td>
  <td>P@5</td>
  <td>P@10</td>
  <td>P@1</td>
  <td>P@5</td>
  <td>P@10</td>
  <td>P@1</td>
  <td>P@5</td>
  <td>P@10</td>
  <td>P@1</td>
  <td>P@5</td>
  <td>P@10</td>
 </tr>
 <tr>
  <td>$L_{style}$</td>
  <td>69.11</td>
  <td>67.86</td>
  <td>67.48</td> 
  <td>64.56</td>
  <td>59.95</td>
  <td>57.26</td>
   
  <td>62.89</td>
  <td>59.58</td>
  <td>58.19</td> 
  <td>57.25</td>
  <td>52.81</td>
  <td>50.43</td>
   
 </tr>
 <tr>
  <td>$L_{style} + L_{triplet}$</td>
  <td>69.71</td>
  <td>68.48</td>
  <td>68.02</td> 
  <td>77.20</td>
  <td>75.80</td>
  <td>75.18</td>
   
  <td>63.04</td>
  <td>60.25</td>
  <td>59.12</td> 
  <td>65.17</td>
  <td>62.65</td>
  <td>61.42</td>
   
 </tr>
 <tr>
  <td>$L_{genre}$</td>
  <td>41.71</td>
  <td>36.79</td>
  <td>34.30</td> 
  <td>77.53</td>
  <td>77.10</td>
  <td>77.18</td>
   
  <td>40.63</td>
  <td>34.61</td>
  <td>31.83</td> 
  <td>67.36</td>
  <td>66.38</td>
  <td>65.93</td>
   
 </tr>
 <tr>
  <td>$L_{genre} + L_{triplet}$</td>
  
  <td>54.82</td>
  <td>51.18</td>
  <td>49.33</td> 
  <td>79.34</td>
  <td>79.02</td>
  <td>78.99</td>
   
  <td>45.96</td>
  <td>40.73</td>
  <td>38.21</td> 
  <td>68.56</td>
  <td>67.62</td>
  <td>67.31</td>
   
 </tr>
 <tr>
  <td>$L_{style} + L_{genre}$</td>
   
  <td>69.09</td>
  <td>66.81</td>
  <td>65.67</td> 
  <td>78.66</td>
  <td>77.41</td>
  <td>76.70</td>
  
  <td>61.79</td>
  <td>57.18</td>
  <td>54.88</td> 
  <td>66.92</td>
  <td>63.99</td>
  <td>62.61</td>
   
 </tr>
 <tr>
  <td>$L_{style} + L_{genre} + L_{triplet}$</td>
  
  <td>69.21</td>
  <td>67.35</td>
  <td>66.49</td> 
  <td>79.83</td>
  <td>79.07</td>
  <td>78.77</td>
   
  <td>61.78</td>
  <td>58.10</td>
  <td>56.15</td> 
  <td>69.17</td>
  <td>67.29</td>
  <td>66.49</td>
 </tr>
</table>

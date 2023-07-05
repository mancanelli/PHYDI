# PHYDI: Initializing Parameterized Hypercomplex Neural Networks as Identity Functions

Matteo Mancanelli, [Eleonora Grassucci](https://sites.google.com/uniroma1.it/eleonoragrassucci/home-page), [Aurelio Uncini](http://www.uncini.com/), and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)

### Abstract

Neural models based on hypercomplex algebra systems are growing and prolificating for a plethora of applications, ranging from computer vision to natural language processing. Hand in hand with their adoption, parameterized hypercomplex neural networks (PHNNs) are growing in size and no techniques have been adopted so far to control their convergence at a large scale. In this paper, we study PHNNs convergence and propose parameterized hypercomplex identity initialization (PHYDI), a method to improve their convergence at different scales, leading to more robust performance when the number of layers scales up, while also reaching the same performance with fewer iterations. We show the effectiveness of this approach in different benchmarks and with common PHNNs with ResNets- and Transformer-based architecture.

### How to use ...


### Cite
Please, cite our work if you found it useful.

```
@inproceedings{mancanelli2023MLSP,
    title={PHYDI: Initializing Parameterized Hypercomplex Neural Networks as Identity Functions},
    author={Mancanelli, Matteo and Grassucci, Eleonora and Barbarossa, Sergio and Comminiello, Danilo},
    year={2023},
    booktitle={IEEE Workshop on Machine Learning for Signal Processing (MLSP)},
}
```

### References 

The code is borrowed and adapted from the following github repo:
1. [HyperNets](https://github.com/eleGAN23/HyperNets)
2. [PHM-Paper-Implementation](https://github.com/MehmetBarutcu/PHM-Paper-Implementation)
3. [ReZero](https://github.com/majumderb/rezero)

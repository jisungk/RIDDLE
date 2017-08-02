![RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning (RIDDLE)](https://user-images.githubusercontent.com/9053987/27894953-4aff74e6-61c4-11e7-901a-8a459026b4ee.png)  
[![Build Status](https://travis-ci.org/jisungk/RIDDLE.svg?branch=master)](https://travis-ci.org/jisungk/RIDDLE) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/jisungk/riddle/blob/master/LICENSE)

**RIDDLE** (**R**ace and ethnicity **I**mputation from **D**isease history with **D**eep **LE**arning) is an open-source Python2 library for using deep learning to impute race and ethnicity information in anonymized electronic medical records (EMRs). RIDDLE provides the ability to (1) build models for estimating race and ethnicity from clinical features, and (2) interpret trained models to describe how specific features contribute to predictions. The RIDDLE library implements the methods introduced in ["RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning"](https://arxiv.org/abs/1707.01623) (arXiv preprint, 2017).

Compared to alternative methods (e.g., scikit-learn/Python, glm/R), RIDDLE is designed to handle large and high-dimensional datasets in a performant fashion. RIDDLE trains models efficiently by running on a parallelized TensorFlow/Theano backend, and avoids memory overflow by preprocessing data in conjunction with batch-wise training.

RIDDLE uses [Keras](https://keras.io) to specify and train the underlying deep neural networks, and [DeepLIFT](https://github.com/kundajelab/deeplift) to compute feature-to-class contribution scores. The current RIDDLE Python module works with both TensorFlow and Theno as the backend to Keras. The default architecture is a deep multi-layer perceptron (deep MLP) that takes binary-encoded features and targets. However, you can specify any neural network architecture (e.g., LSTM, CNN) and data format by writing your own `model_module` files! 

### Documentation
Please visit [riddle.ai](https://riddle.ai).

### Dependencies  
Python Libraries:  
* Keras (`keras`)  
* DeepLIFT (`deeplift`, install from GitHub)
* TensorFlow (`tensorflow`) or Theano (`theano`)  
* scikit-learn (`sklearn`)  
* NumPy (`numpy`)  
* SciPy (`scipy`)  
* Matplotlib (`matplotlib`)  
* h5py (`h5py`)  

General:  
* HDF5

### Unit testing
Execute the following command in the outer *repository* folder (not `riddle/riddle`):
```
% PYTHONPATH=. pytest
```

### FAQ

#### What's the easiest way to install RIDDLE?

You can clone the GitHub repo and go from there:
```
% git clone --recursive git://github.com/jisungk/riddle.git
```

Alternatively, you can install RIDDLE and DeepLIFT from GitHub using `pip`:
```
% pip install git+git://github.com/kundajelab/deeplift.git # DeepLIFT
% pip install git+git://github.com/jisungk/riddle.git      # RIDDLE
```

#### What is the default format for data files?

Please refer to the example data file `dummy.txt` and the accompanying `README` in the [`_data` directory](https://github.com/jisungk/riddle/tree/master/_data).

### Authors

[Ji-Sung Kim](http://jisungkim.com)  
Princeton University  
*hello (at) jisungkim.com*

[Andrey Rzhetsky](https://scholar.google.com/citations?user=HXCMYLsAAAAJ&hl=en), Edna K. Papazian Professor  
University of Chicago  
*andrey.rzhetsky (at) uchicago.edu*

### License & Attribution
All media (including but not limited to designs, images and logos) are copyrighted by Ji-Sung Kim (2017). 

Project code (explicitly excluding media) is licensed under the Apache License 2.0. If you would like use or modify this project or any code presented here, please include the [notice](https://github.com/jisungk/riddle/NOTICE) and [license](https://github.com/jisungk/riddle/LICENSE) files, and cite: 
```
@article{KimJS2017RIDDLE,
  title={RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning},
  author={Kim, Ji-Sung and Rzhetsky, Andrey},
  journal={arXiv preprint arXiv:1707.01623},
  year={2017}
}
```
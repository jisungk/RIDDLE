![RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning (RIDDLE)](https://user-images.githubusercontent.com/9053987/27894953-4aff74e6-61c4-11e7-901a-8a459026b4ee.png)
[![Build Status](https://travis-ci.org/jisungk/RIDDLE.svg?branch=master)](https://travis-ci.org/jisungk/RIDDLE)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/jisungk/riddle/blob/master/LICENSE)

**RIDDLE** (**R**ace and ethnicity **I**mputation from **D**isease history with **D**eep **LE**arning) is an open-source Python2 library for using deep learning to impute race and ethnicity information in anonymized electronic medical records (EMRs). RIDDLE provides the ability to (1) build models for estimating race and ethnicity from clinical features, and (2) interpret trained models to describe how specific features contribute to predictions. The RIDDLE library implements the methods introduced in ["RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning"](https://doi.org/10.1371/journal.pcbi.1006106) (PLOS Computational Biology, 2018).

Compared to alternative methods (e.g., scikit-learn/Python, glm/R), RIDDLE is designed to handle large and high-dimensional datasets in a performant fashion. RIDDLE trains models efficiently by running on a parallelized TensorFlow/Theano backend, and avoids memory overflow by preprocessing data in conjunction with batch-wise training.

RIDDLE uses [Keras](https://keras.io) to specify and train the underlying deep neural networks, and [DeepLIFT](https://github.com/kundajelab/deeplift) to compute feature-to-class contribution scores. The current RIDDLE Python module works with both TensorFlow and Theno as the backend to Keras. The default architecture is a deep multi-layer perceptron (deep MLP) that takes binary-encoded features and targets. However, you can specify any neural network architecture (e.g., LSTM, CNN) and data format by writing your own `model_module` files!

### Documentation
Please visit [riddle.ai](https://riddle.ai).

### Dependencies
Python Libraries:
* Keras (`keras`)
* DeepLIFT (`deeplift`, available on GitHub)
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
% cd riddle
% pip install -r requirements.txt
```

#### How can I run the RIDDLE pipeline?

Execute the following scripts.
```
% python parameter_search.py  # run parameter tuning
% python riddle.py            # train and evaluate the model
% python interpret_riddle.py  # interpret the traiend model
```

#### What is the default format for data files?

Please refer to the example data file `dummy.txt` and the accompanying `README` in the [`_data` directory](https://github.com/jisungk/riddle/tree/master/_data).

### Authors

[Ji-Sung Kim](http://jisungkim.com)  
Princeton University  
*hello (at) jisungkim.com* (technical inquiries)  

[Xin Gao](https://scholar.google.com/citations?user=wqdK8ugAAAAJ&hl=en), Associate Professor  
King Abdullah University of Science and Technology  

[Andrey Rzhetsky](https://scholar.google.com/citations?user=HXCMYLsAAAAJ&hl=en), Edna K. Papazian Professor  
University of Chicago  
*andrey.rzhetsky (at) uchicago.edu* (research inquiries)  

### License & Attribution
All media (including but not limited to designs, images and logos) are copyrighted by Ji-Sung Kim (2017).

Project code (explicitly excluding media) is licensed under the Apache License 2.0. If you would like use or modify this project or any code presented here, please include the [notice](https://github.com/jisungk/riddle/NOTICE) and [license](https://github.com/jisungk/riddle/LICENSE) files, and cite:

```
@article{10.1371/journal.pcbi.1006106,
    author = {Kim, Ji-Sung AND Gao, Xin AND Rzhetsky, Andrey},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning},
    year = {2018},
    month = {04},
    volume = {14},
    url = {https://doi.org/10.1371/journal.pcbi.1006106},
    pages = {1-15},
    number = {4},
    doi = {10.1371/journal.pcbi.1006106}
}
```

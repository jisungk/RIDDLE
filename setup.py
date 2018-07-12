from setuptools import setup, find_packages

long_description = 'RIDDLE (Race and ethnicity Imputation from Disease history with Deep LEarning) is an open-source deep learning (DL) framework for estimating/imputing race and ethnicity information in anonymized electronic medical records (EMRs). It utilizes Keras, a modular DL library, and DeepLIFT, an algorithm by Shrikumar et al. (2016) for learning important features in deep neural networks. ' + \
    'Please see the PLOS Computational Biology paper (https://doi.org/10.1371/journal.pcbi.1006106) for information on the research project results and design. \n' + \
    'The riddle Python 2 library makes it easy to perform categorical imputations using a variety of DL architectures -- not just for EMR datasets. Furthermore, compared to alternative methods (e.g., scikit-learn/Python, Amelia II/R), RIDDLE is more efficient due to its parallelized backend (TensorFlow under Keras). ' + \
    'RIDDLE uses Keras to specify, train, and build the underlying DL models. It was debugged using Keras with a TensorFlow backend. The default architecture is a deep multilayer perceptron (deep MLP) that takes "one-hot-encoded" features. However, you can specify any DL architecture (e.g., LSTM, CNN) by writing your own model_module files! '

setup(
    name='RIDDLE',
    version='2.0.1',
    description='Race and ethnicity Imputation from Disease history with Deep LEarning',
    long_description=long_description,
    author='Ji-Sung Kim',
    author_email='hello (at) jisungkim.com',
    url='https://riddle.ai',
    license='Apache 2.0',
    download_url='https://github.com/jisungk/riddle/archive/master.tar.gz',
    packages=find_packages(exclude=['tests*']),
    install_requires=['keras', 'tensorflow', 'sklearn', 'xgboost', 'numpy',
                      'scipy', 'matplotlib', 'h5py'],
    keywords=['deep learning', 'machine learning', 'neural networks',
              'imputation', 'emr', 'epidemiology', 'biomedicine', 'biology',
              'computational bioloigy', 'bioinformatics']
)

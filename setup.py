from setuptools import setup, find_packages

long_description = 'RIDDLE (Race and ethnicity Imputation from Disease with Deep LEarning) is an open-source deep learning (DL) framework for estimating/imputing race and ethnicity information in anonymized electronic medical records (EMRs). It utilizes Keras, a modular DL library, and deepLIFT, an algorithm by Shrikumar et al. (2016) for learning important features in deep neural networks.\n' + \
'The riddle Python2 library makes it easy to perform categorical imputations using a variety of DL architectures – not just for EMR datasets. Furthermore, compared to alternative methods (e.g., scikit-learn/Python, Amelia II/R), RIDDLE is remarkably efficient due to its parallelized backends (TensorFlow, Theano).' + \
'RIDDLE uses Keras to specify, train, and build the underlying DL models. The current riddle module works with both TensorFlow and Theno as the backend to Keras. However, the requisite deeplift package does not formally support models created using Keras with a TensorFlow backend. The default architecture is a deep multi-layer perceptron (deep MLP) that takes “one-hot-encoded” features. However, you can specify any DL architecture (e.g., LSTM, CNN) by writing your own model_module files!'

setup(
    name='RIDDLE',
    version='1.0.0',
    description='Race and ethnicity Imputation from Disease with Deep LEarning',
    long_description=long_description,
    author='Ji-Sung Kim',
    author_email='jisungk (at) cs.princeton.edu',
    url='https://riddle.ai',
    download_url='https://github.com/jisungk/riddle/tarball/1.0.0',
    license='-',
    packages=find_packages(exclude=['tests*']),
    install_requires=['keras', 'tensorflow', 'sklearn', 'numpy', 'scipy', 
        'h5py'],
    keywords = ['deep learning', 'machine learning', 'neural networks' 'imputation', 'emr', 'epidemiology', 'biomedicine', 'biology', 'computational bioloigy', 'bioinformatics']
)
from setuptools import setup, find_packages

long_description = 'RIDDLE (Race and ethnicity Imputation from Disease ' + \
'history with Deep LEarning) is an open-source deep learning (DL) ' + \
'framework for estimating/imputing race and ethnicity information in ' + \
'anonymized electronic medical records (EMRs). It utilizes Keras, a ' + \
'modular DL library, and DeepLIFT, an algorithm by Shrikumar et al. (2016)' + \
'for learning important features in deep neural networks.\n' + \
'The RIDDLE Python2 library makes it easy to perform similar categorical ' + \
'imputations using a variety of DL architectures – not just for EMR ' + \
'datasets. Furthermore, compared to alternative methods (e.g., ' + \
'scikit-learn/Python, Amelia II/R), RIDDLE is remarkably efficient due to ' + \
'its parallelized backends (TensorFlow, Theano). RIDDLE uses Keras to ' + \
'specify, train, and build the underlying DL models. The default ' + \
'architecture is a deep multi-layer perceptron (deep MLP) that takes ' + \
'binary-encoded features. However, you can specify any DL architecture ' + \
'(e.g., LSTM, CNN) by writing your own `model_module` files!'

setup(
    name='RIDDLE',
    version='1.0.0',
    description='Race and ethnicity Imputation from Disease history with Deep LEarning',
    long_description=long_description,
    author='Ji-Sung Kim',
    author_email='jisungk (at) cs.princeton.edu',
    url='https://riddle.ai',
    license='Apache 2.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=['keras', 'tensorflow', 'sklearn', 'numpy', 'scipy', 
        'matplotlib', 'h5py'],
    keywords = ['deep learning', 'machine learning', 'neural networks', 
        'imputation', 'emr', 'epidemiology', 'biomedicine', 'biology', 
        'computational bioloigy', 'bioinformatics']
)
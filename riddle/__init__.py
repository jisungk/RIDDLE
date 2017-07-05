from __future__ import absolute_import

from . import emr
from . import feature_importance
from . import ordering
from . import frequency
from . import roc
from . import utils
from . import models
from . import parameter_tuning

__version__ = '1.0.0'

def hello():
    print('Hello, World')
    print('My name is RIDDLE {}'.format(__version__))
"""RIDDLE module (Python 2.7+)."""

from __future__ import absolute_import
from __future__ import print_function

from . import emr
from . import feature_importance
from . import frequency
from . import models
from . import ordering
from . import roc
from . import tuning

__version__ = '2.0.1'


def hello():
    """Print out the current version."""
    print('Hello, World')
    print('My name is RIDDLE {}'.format(__version__))

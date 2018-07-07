"""summary.py

Provides a Summary class for aggregating list information.

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function
from collections import OrderedDict
import sys


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


class Summary(object):
    """Helper class which represents some summary.

    Stores lists using an OrderedDictionary to maintain the order of list items.
    All items must be equal length.
    """

    def __init__(self, ordered_dict):
        """Initialize Summary object."""
        assert isinstance(ordered_dict, OrderedDict)

        if not len(set([len(val) for val in ordered_dict.values()])) == 1:
            for key, l in [(key, len(val)) for key, val in ordered_dict.items()]:
                eprint(key, l)
            raise ValueError('Different size lists in summary.')

        self.od = ordered_dict

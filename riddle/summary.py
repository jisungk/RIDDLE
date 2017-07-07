'''
summary.py

Provides a Summary class for aggregating list information. 

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
'''

from __future__ import print_function
import sys

from .utils import eprint

'''
Helper class which stores lists using an OrderedDictionary to maintain the order
of list items. All items must be of equal length. 
'''
class Summary():
    def __init__(self, ordered_dict):
        if not len(set([len(val) for val in ordered_dict.values()])) == 1: 
            for key, l in [(key, len(val)) for key, val in ordered_dict.items()]: 
                eprint(key, l) 
            raise AssertionError('Different size lists in summary') 

        self.od = ordered_dict
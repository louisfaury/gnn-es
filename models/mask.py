#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:53:27 2018
BinaryMask class file.
@author: l.faury
"""

import numpy as np


class BinaryMask(object):
    """ Binary mask class """
    def __init__(self, size):
        self.size = size

    def __call__(self):
        raise NotImplementedError(str(type(self))+' does not implement __call__')


class EvenMask(BinaryMask):
    """ 1 for even, 0 for odds"""
    def __init__(self, size):
        super().__init__(size)

    def __call__(self):
        return np.array([1 if i % 2 == 0 else 0 for i in range(self.size)])


class OddMask(BinaryMask):
    """ 0 for even, 1 for odds """
    def __init__(self, size):
        super().__init__(size)

    def __call__(self):
        return 1 - np.array([1 if i % 2 == 0 else 0 for i in range(self.size)])


class RandomMask(BinaryMask):
    """ Randomly drawn 0's and 1's """
    def __init__(self, size, positive=None):
        super().__init__(size)
        if positive is None:
            self.positive = np.random.choice(self.size, int(self.size/2), replace=False)
        else:
            self.positive = positive

    def __call__(self):
        return np.array([1 if i in self.positive else 0 for i in range(self.size)])

    def inverse(self):
        return np.array([i for i in range(self.size) if i not in self.positive])


if __name__ == '__main__':
    print('Testing Binary and Odd masks')
    emask = EvenMask(10)
    omask = OddMask(10)
    print(emask())
    print(omask())

    print('Testing random masks')
    rmask = RandomMask(10)
    rinvmask = RandomMask(10, rmask.inverse())
    print(rmask())
    print(rinvmask())

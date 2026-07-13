import numpy as np
import scipy
from ctypes import *
import sys, os, glob
from typing import Any, Callable
from tqdm import tqdm

class LeastSquares:

    def __init__(self, *args, **kwargs) -> Any:
        for _, __ in kwargs: self.__setattr__(_, __)
        for _ in args: print(f'Attr {_} not set ')

    def __repr__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

    
    def call(self, **kwargs) -> ...:...
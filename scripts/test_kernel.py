# Minimal executed test notebook
nbformat = None
src = '''# %matplotlib inline
print("hello kernel")
print(1 + 1)
import numpy as np
print(np.__version__)
'''
print(src)

import pyximport
import numpy as np
from distutils.extension import Extension
sources = Extension("binding", ["binding.pyx", "fft.c"], include_dirs=[np.get_include()], libraries=["m"])
pyximport.install(setup_args={'ext_modules':[sources]})

from binding import fft

r, i = fft(np.arange(8, dtype=np.float))
print r
print i

z = np.fft.fft(np.arange(8))
zr = np.vectorize(lambda x : x.real)(z)
zi = np.vectorize(lambda x : x.imag)(z)

print zr
print zi

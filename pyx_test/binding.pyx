import numpy as np
cimport numpy as np

cimport cython

cdef extern:
    size_t _fft2(double *x, size_t N, int s, double *r, double *i)

def fft(x):
    cdef np.ndarray[double] x_ = x
    cdef np.ndarray[double] r = np.zeros(x.shape, dtype=np.float)
    cdef np.ndarray[double] i = np.zeros(x.shape, dtype=np.float)

    _fft2(&x_[0], x.shape[0], 1, &r[0], &i[0])

    return r,i

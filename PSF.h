#ifndef PSF_UTILS_H
#define PSF_UTILS_H

#include <fftw3.h>

fftw_complex* PSF_Transform(int L, int M, int N, double* PSF);

fftwf_complex* PSF_Transform_Float(int L, int M, int N, float* PSF);

#endif // PSF_UTILS_H

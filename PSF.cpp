#include <iostream>
#include <fftw3.h>
#include <chrono>
#include <omp.h>
#include "PlanCreate.h"

fftw_complex* PSF_Transform(int L, int M, int N, double* PSF){

    fftw_complex* PSF_FT = (fftw_complex*)fftw_alloc_complex(L*M*(N/2 + 1));
    fftw_plan FFT2d_PSF = create_batched_2d_fft_plan(L, M, N, PSF, PSF_FT);

    fftw_execute(FFT2d_PSF);

    return PSF_FT;
}

fftwf_complex* PSF_Transform_Float(int L, int M, int N, float* PSF){

    fftwf_complex* PSF_FT = (fftwf_complex*)fftwf_alloc_complex(L*M*(N/2 + 1));
    fftwf_plan FFT2d_PSF = create_batched_2d_fft_plan_float(L, M, N, PSF, PSF_FT);

    fftwf_execute(FFT2d_PSF);

    return PSF_FT;
}
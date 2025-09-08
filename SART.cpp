#include <iostream>
#include <fftw3.h>
#include <chrono>
#include "PlanCreate.h"
#include "IterOps.h"
#include "Project.h"
#include "IterOpsSIMD.h"
#include "PSF.h"
#include <omp.h>


// AS IMPLEMENTED THIS IS NOT A 'SART' ALGORITHM
// IT IS MISSING THE NORMALIZATION STEP USED IN SART
// THIS IS CLOSER TO A PURE GRADIENT DESCENT METHOD
extern "C" {
void SART(
    int L, // Number of Depths
    int M, // Number of Rows
    int N, // Number of Columns
    double (*Vol), // L*M*N array of Volume Intensity Guess
    double (*PSF), // L*M*N array of PSFs
    double (*IMG), // M*N array of Observed Intensity
    const int ITERS  // Number of RLD iterations
)
{
    fftw_init_threads();
    int Nthreads = omp_get_max_threads();
    omp_set_num_threads(Nthreads);
    fftw_plan_with_nthreads(Nthreads);

    fftw_import_wisdom_from_filename("fftw_wisdom.dat");

    int ImgSize = M*N;
    int TotalVolSize = L*ImgSize;
    int Nfft = (N/2 + 1);

    double* Vol_Temp = (double*)fftw_alloc_real(L*M*N);                 // Volumetric Intensities Temp
    fftw_complex* Vol_FT = (fftw_complex*)fftw_alloc_complex(L*M*Nfft); // 2D Fourier Transform of Volume Slices

    double* FP = (double*)fftw_alloc_real(M*N);                         // Forward Projection
    fftw_complex* FP_FT = (fftw_complex*)fftw_alloc_complex(M*Nfft);    // Forward Projection Fourier Transform

    fftw_complex* PSF_FT = PSF_Transform(L, M, N, PSF);                 // Fourier Transform of PSFs

    fftw_plan FFT2d_Forward_Batch = create_batched_2d_fft_plan(L, M, N, Vol, Vol_FT);         // Volume -> Volume FT
    fftw_plan FFT2d_Backward = fftw_plan_dft_c2r_2d(M, N, Vol_FT, FP, FFTW_ESTIMATE);          // First Slice of Volume FT -> FP

    fftw_plan FFT2d_Forward = fftw_plan_dft_r2c_2d(M, N, FP, FP_FT, FFTW_ESTIMATE);            // IMG / FP -> FP_FT
    fftw_plan FFT2d_Backward_Batch = create_batched_2d_ifft_plan(L, M, N, Vol_FT, Vol_Temp);  // Volume FT -> Vol_Temp

    std::cout << "SART Progress: " << std::endl;
    std::cout << "Iter 0 / " << ITERS;

    for (int iter = 0; iter < ITERS; iter++){

        ProjForward(L, M, N, FFT2d_Forward_Batch, Vol_FT, PSF_FT, FFT2d_Backward);

        ElementWiseDifference_InPlace_Double(M*N, FP, IMG);

        ProjBackward(L, M, N, FFT2d_Forward, FP_FT, PSF_FT, Vol_FT, FFT2d_Backward_Batch);

        ElementWiseAdd_InPlace_Double(L*M*N, Vol, Vol_Temp);

        std::cout << "\rIter " << (iter+1) << "/ " << ITERS << "     ";

    }

    std::cout << std::endl;


    // Free allocated memory and destroy FFTW plans
    fftw_free(Vol_Temp);
    fftw_free(Vol_FT);

    fftw_free(FP);
    fftw_free(FP_FT);

    fftw_free(PSF_FT);

    fftw_destroy_plan(FFT2d_Forward_Batch);
    fftw_destroy_plan(FFT2d_Backward);
    fftw_destroy_plan(FFT2d_Forward);
    fftw_destroy_plan(FFT2d_Backward_Batch);


    fftw_export_wisdom_to_filename("fftw_wisdom.dat");
    return;
}
}

extern "C" {
void SARTf(
    int L, // Number of Depths
    int M, // Number of Rows
    int N, // Number of Columns
    float (*Vol), // L*M*N array of Volume Intensity Guess
    float (*PSF), // L*M*N array of PSFs
    float (*IMG), // M*N array of Observed Intensity
    const int ITERS  // Number of RLD iterations
)
{
    fftwf_init_threads();
    int Nthreads = omp_get_max_threads();
    omp_set_num_threads(Nthreads);
    fftwf_plan_with_nthreads(Nthreads);

    fftwf_import_wisdom_from_filename("fftwf_wisdom.dat");

    int ImgSize = M*N;
    int TotalVolSize = L*ImgSize;
    int Nfft = (N/2 + 1);

    float* Vol_Temp = (float*)fftwf_alloc_real(L*M*N);                 // Volumetric Intensities Temp
    fftwf_complex* Vol_FT = (fftwf_complex*)fftwf_alloc_complex(L*M*Nfft); // 2D Fourier Transform of Volume Slices

    float* FP = (float*)fftwf_alloc_real(M*N);                         // Forward Projection
    fftwf_complex* FP_FT = (fftwf_complex*)fftwf_alloc_complex(M*Nfft);    // Forward Projection Fourier Transform

    fftwf_complex* PSF_FT = PSF_Transform_Float(L, M, N, PSF);                 // Fourier Transform of PSFs

    fftwf_plan FFT2d_Forward_Batch = create_batched_2d_fft_plan_float(L, M, N, Vol, Vol_FT);         // Volume -> Volume FT
    fftwf_plan FFT2d_Backward = fftwf_plan_dft_c2r_2d(M, N, Vol_FT, FP, FFTW_ESTIMATE);          // First Slice of Volume FT -> FP

    fftwf_plan FFT2d_Forward = fftwf_plan_dft_r2c_2d(M, N, FP, FP_FT, FFTW_ESTIMATE);            // IMG / FP -> FP_FT
    fftwf_plan FFT2d_Backward_Batch = create_batched_2d_ifft_plan_float(L, M, N, Vol_FT, Vol_Temp);  // Volume FT -> Vol_Temp

    std::cout << "SART Progress: " << std::endl;
    std::cout << "Iter 0 / " << ITERS;

    for (int iter = 0; iter < ITERS; iter++){

        ProjForward_Float(L, M, N, FFT2d_Forward_Batch, Vol_FT, PSF_FT, FFT2d_Backward);

        ElementWiseDifference_InPlace_Float(M*N, FP, IMG);

        ProjBackward_Float(L, M, N, FFT2d_Forward, FP_FT, PSF_FT, Vol_FT, FFT2d_Backward_Batch);

        ElementWiseAdd_InPlace_Float(L*M*N, Vol, Vol_Temp);

        std::cout << "\rIter " << (iter+1) << "/ " << ITERS << "     ";

    }

    std::cout << std::endl;

    fftwf_free(Vol_Temp);
    fftwf_free(Vol_FT);

    fftwf_free(FP);
    fftwf_free(FP_FT);

    fftwf_free(PSF_FT);

    fftwf_destroy_plan(FFT2d_Forward_Batch);
    fftwf_destroy_plan(FFT2d_Backward);
    fftwf_destroy_plan(FFT2d_Forward);
    fftwf_destroy_plan(FFT2d_Backward_Batch);


    fftwf_export_wisdom_to_filename("fftwf_wisdom.dat");
    return;
}
}

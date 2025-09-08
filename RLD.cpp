#include <iostream>
#include <fftw3.h>
#include <chrono>
#include "PlanCreate.h"
#include "IterOps.h"
#include "Project.h"
#include "IterOpsSIMD.h"
#include "PSF.h"
#include <omp.h>

extern "C" {
void RLD(
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

    std::cout << "RLD Progress: " << std::endl;
    std::cout << "Iter 0 / " << ITERS;

    for (int iter = 0; iter < ITERS; iter++){

        ProjForward(L, M, N, FFT2d_Forward_Batch, Vol_FT, PSF_FT, FFT2d_Backward);

        ElementWiseDivide_InPlace_Double(M, N, FP, IMG);

        ProjBackward(L, M, N, FFT2d_Forward, FP_FT, PSF_FT, Vol_FT, FFT2d_Backward_Batch);

        ElementWiseMult_InPlace_Double(L, M, N, Vol, Vol_Temp);

        std::cout << "\rIter " << (iter+1) << "/ " << ITERS << "     ";

    }

    std::cout << std::endl;

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
void RLDf(
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

    std::cout << "RLD Progress: " << std::endl;
    std::cout << "Iter 0 / " << ITERS;

    for (int iter = 0; iter < ITERS; iter++){

        ProjForward_Float(L, M, N, FFT2d_Forward_Batch, Vol_FT, PSF_FT, FFT2d_Backward);

        ElementWiseDivide_InPlace_Float(M, N, FP, IMG);

        ProjBackward_Float(L, M, N, FFT2d_Forward, FP_FT, PSF_FT, Vol_FT, FFT2d_Backward_Batch);

        ElementWiseMult_InPlace_Float(L, M, N, Vol, Vol_Temp);

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

int main(){
    std::cout << "RLD Init" << std::endl;

    int L = 32;
    int M = 256;
    int N = 256;
    // int M = 5120;
    // int N = 6400;

    double* psf = (double*)fftw_alloc_real(L*M*N);
    double* vol = (double*)fftw_alloc_real(L*M*N);
    double* img0 = (double*)fftw_alloc_real(M*N);

    for (int i = 0; i < (L*M*N); i++){
        psf[i] = 0.0;
        vol[i] = 1.0;
    }

    for (int i = 0; i < (M*N); i++){
        img0[i] = 0.0;
    }

    // Loop to set specific values as per the given Python code
    for (int i = 0; i < L; ++i) {
        psf[i * M * N + 0 * N + 0] = 0.25;
        psf[i * M * N + 0 * N + (N - (i + 1))] = 0.25;
        psf[i * M * N + (M - (i + 1)) * N + 0] = 0.25;
        psf[i * M * N + (M - (i + 1)) * N + (N - (i + 1))] = 0.25;

        img0[0 * N + 0] += 0.25;
        img0[0 * N + (N - (i + 1))] += 0.25;
        img0[(M - (i + 1)) * N + 0] += 0.25;
        img0[(M - (i + 1)) * N + (N - (i + 1))] += 0.25;
    }

    // Optional: Print the array to verify the values

    // std::cout << "Img0" << std::endl;
    for (int j = 0; j < M; ++j) {
        // std::cout << std::endl;
        for (int k = 0; k < N; ++k) {
            // std::cout << img0[j * N + k] << ", ";
        }
    }

    // std::cout << "PSF" << std::endl;

    // Optional: Print the array to verify the values
    for (int i = 0; i < L; ++i) {
        // std::cout << std::endl;
        for (int j = 0; j < M; ++j) {
            // std::cout << std::endl;
            for (int k = 0; k < N; ++k) {
                // std::cout << psf[i * M * N + j * N + k] << ", ";
            }
        }
    }

    const int ITERS = 5;

    fftw_init_threads();
    int Nthreads = omp_get_max_threads();
    omp_set_num_threads(Nthreads);
    fftw_plan_with_nthreads(Nthreads);
    std:: cout << "Number of Threads: " << Nthreads << std::endl; 

    fftw_import_wisdom_from_filename("fftw_wisdom.dat");

    auto start = std::chrono::high_resolution_clock::now();
    RLD(L, M, N, vol, psf, img0, ITERS);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "RLD Time: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // RLD_SIMD(L, M, N, vol, psf, img0, ITERS);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "RLD w/ SIMD Time: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    RLD(L, M, N, vol, psf, img0, ITERS);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "RLD Time: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // RLD_SIMD(L, M, N, vol, psf, img0, ITERS);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "RLD w/ SIMD Time: " << duration.count() << " seconds" << std::endl;

    fftw_export_wisdom_to_filename("fftw_wisdom.dat");

    // std::cout << std::endl << "Vol";

    //     // Optional: Print the array to verify the values
    // for (int i = 0; i < L; ++i) {
    //     std::cout << std::endl;
    //     for (int j = 0; j < M; ++j) {
    //         std::cout << std::endl;
    //         for (int k = 0; k < N; ++k) {
    //             std::cout << vol[i * M * N + j * N + k] << ", ";
    //         }
    //     }
    // }
    fftw_free(vol);
    fftw_free(psf);
    fftw_free(img0);

    return 0;
}


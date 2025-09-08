#ifndef FFTW_UTILS_H
#define FFTW_UTILS_H

#include <fftw3.h>

// Function declarations

/**
 * @brief Creates a batched 2D FFT plan for real-to-complex transforms.
 * 
 * @param L The number of 2D transforms (batch size).
 * @param M The number of rows in each 2D transform.
 * @param N The number of columns in each 2D transform.
 * @param real_data Pointer to the input array containing real data.
 * @param complex_data Pointer to the output array for complex data.
 * @return fftw_plan The FFTW plan object for the batched 2D FFTs.
 */
fftw_plan create_batched_2d_fft_plan(int L, int M, int N, double* real_data, fftw_complex* complex_data);

/**
 * @brief Creates a batched 2D IFFT plan for complex-to-real transforms.
 * 
 * @param L The number of 2D transforms (batch size).
 * @param M The number of rows in each 2D transform.
 * @param N The number of columns in each 2D transform.
 * @param complex_data Pointer to the input array containing complex data.
 * @param real_data Pointer to the output array for real data.
 * @return fftw_plan The FFTW plan object for the batched 2D IFFTs.
 */
fftw_plan create_batched_2d_ifft_plan(int L, int M, int N, fftw_complex* complex_data, double* real_data);

/**
 * @brief Creates a batched 2D FFT plan for real-to-complex transforms.
 * 
 * @param L The number of 2D transforms (batch size).
 * @param M The number of rows in each 2D transform.
 * @param N The number of columns in each 2D transform.
 * @param real_data Pointer to the input array containing real data.
 * @param complex_data Pointer to the output array for complex data.
 * @return fftw_plan The FFTW plan object for the batched 2D FFTs.
 */
fftwf_plan create_batched_2d_fft_plan_float(int L, int M, int N, float* real_data, fftwf_complex* complex_data);

/**
 * @brief Creates a batched 2D IFFT plan for complex-to-real transforms.
 * 
 * @param L The number of 2D transforms (batch size).
 * @param M The number of rows in each 2D transform.
 * @param N The number of columns in each 2D transform.
 * @param complex_data Pointer to the input array containing complex data.
 * @param real_data Pointer to the output array for real data.
 * @return fftw_plan The FFTW plan object for the batched 2D IFFTs.
 */
fftwf_plan create_batched_2d_ifft_plan_float(int L, int M, int N, fftwf_complex* complex_data, float* real_data);

#endif // FFTW_UTILS_H
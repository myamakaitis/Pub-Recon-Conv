// I attemted to use SIMD intrinsics to speed up some of the iterative operations
// My implementations did not seem to be faster than compiler auto-vectorization
// so I have not used these functions in the main codebase

#ifndef ITEROPS_SIMD_H
#define ITEROPS_SIMD_H

#include <immintrin.h> // Include AVX2 intrinsics header
#include <fftw3.h> // Include FFTW3 library

// SIMD optimized function to sum along the first axis of a 3D array in-place for complex FFTW data
void SumAxis0_InPlace_FFTW_Complex_SIMD(int L, int M, int N, fftw_complex* complex_data);

// SIMD optimized function for element-wise multiplication of double arrays
void ElementWiseMult_InPlace_Double_SIMD(int M, int N, int L, double* VolMod, double* VolMult);

// SIMD optimized function for element-wise multiplication of FFTW complex arrays
void ElementWiseMult_InPlace_FFTW_Complex_SIMD(int L, int M, int N, fftw_complex* VolWrite, fftw_complex* VolRead);

// SIMD optimized function for element-wise division of double arrays
void ElementWiseDivide_InPlace_Double_SIMD(int M, int N, double* fp, double* img);

#endif // ITEROPS_SIMD_H
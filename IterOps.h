#ifndef ITEROPS_H
#define ITEROPS_H

#include <fftw3.h>

// Function declarations

// Performs in-place summation along the first axis of a 3D array (represented as a 1D array) 
// and normalizes each summed element by dividing by the product of the second and third dimensions.
// void SumAxis0_InPlace_Normalize(int L, int M, int N, double* real_data);

// Performs in-place summation along the first axis of a 3D array of complex numbers
// (represented as fftw_complex) and normalizes each summed element.
void SumAxis0_InPlace_Normalize_FFTW_Complex(int L, int M, int N, fftw_complex* complex_data);

void ElementWiseAdd_InPlace_Double(int N, double* VolMod, double* VolAdd);

void ElementWiseDifference_InPlace_Double(int N, double* fp, double* img);

void ElementWiseLog2_InPlace_Double(int N, double* array, double offset);

void ElementWiseExp2_InPlace_Double(int N, double* array, double offset);

// Performs element-wise multiplication of two 3D arrays of double values.
// The result is stored in the first array, modifying it in place.
void ElementWiseMult_InPlace_Double(int M, int N, int L, double* VolMod, double* VolMult);

// Performs element-wise division of two 2D arrays of double values.
// The result is stored in the first array, modifying it in place.
void ElementWiseDivide_InPlace_Double(int M, int N, double* fp, double* img);

// Performs element-wise multiplication of two 3D arrays of complex numbers (fftw_complex).
// The result is stored in the first array, modifying it in place.
void ElementWiseMult_InPlace_FFTW_Complex(int TotalSize, fftw_complex* VolWrite, fftw_complex* VolRead);

// Performs element-wise multiplication with conjugation on complex arrays.
// The result is stored in the destination array, modifying it in place.
void ElementwiseMultConjugate_FFTW_Complex(int L, int M, int N, fftw_complex* dest, fftw_complex* VolRatio_FT, fftw_complex* PSF_FT);

// Performs in-place summation along the first axis of a 3D array of complex numbers
// (represented as fftw_complex) and normalizes each summed element.
void SumAxis0_InPlace_Normalize_FFTW_ComplexF(int L, int M, int N, fftwf_complex* complex_data);

void ElementWiseAdd_InPlace_Float(int N, float* VolMod, float* VolAdd);

void ElementWiseDifference_InPlace_Float(int N, float* fp, float* img);

void ElementWiseLog2_InPlace_Float(int N, float* array, float offset);

void ElementWiseExp2_InPlace_Float(int N, float* array, float offset);

// Performs element-wise multiplication of two 3D arrays of double values.
// The result is stored in the first array, modifying it in place.
void ElementWiseMult_InPlace_Float(int M, int N, int L, float* VolMod, float* VolMult);

// Performs element-wise division of two 2D arrays of double values.
// The result is stored in the first array, modifying it in place.
void ElementWiseDivide_InPlace_Float(int M, int N, float* fp, float* img);

// Performs element-wise multiplication of two 3D arrays of complex numbers (fftw_complex).
// The result is stored in the first array, modifying it in place.
void ElementWiseMult_InPlace_FFTW_ComplexF(int TotalSize, fftwf_complex* VolWrite, fftwf_complex* VolRead);

// Performs element-wise multiplication with conjugation on complex arrays.
// The result is stored in the destination array, modifying it in place.
void ElementwiseMultConjugate_FFTW_ComplexF(int L, int M, int N, fftwf_complex* dest, fftwf_complex* VolRatio_FT, fftwf_complex* PSF_FT);

#endif // ITEROPS_H
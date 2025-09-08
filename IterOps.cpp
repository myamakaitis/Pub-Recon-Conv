#include <cmath>
#include <fftw3.h>
#include <iostream>
#include <omp.h>
#include <limits>
// Function: SumAxis0_InPlace_Normalize
// Description: This function performs an in-place summation along the first axis (axis 0) 
//              of a 3D array represented as a 1D array in row-major order. It then normalizes 
//              each summed element by dividing by the product of the second and third dimensions (M * N).
// 
// Parameters:
// - L: The size of the first dimension (number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - real_data: Pointer to the 1D array of doubles representing the 3D array of size L x M x N.
void SumAxis0_InPlace_Normalize(int L, int M, int N, double* real_data) {
    int mn_index = 0;     // Index for accessing the elements in the 2D plane (M x N)    
    int MN = (M * N);    // MN premultiplied
    int L_index = MN;    // Loop Index for axis 0
    // double MN_inv = 1.0; // /(MN);   // The normalization factor to scale the summed elements by the size of each 2D slice
    
    // #pragma omp parallel for
    // Loop over each row (M) of the 2D slice
    for (int j = 0; j < M; j++) {
        // Loop over each column (N) of the current row
        for (int i = 0; i < N; i++) {
            // Loop over each layer (L), summing along the first axis (axis 0)
            for (int k = 1; k < L; k++) {         
                // Add the element from the current layer (k) to the base layer (k = 0)
                real_data[mn_index] = real_data[mn_index] + real_data[mn_index + L_index];
                L_index += MN;
            }

            // Normalize the summed value by dividing by the size of the 2D slice (M * N)
            // real_data[mn_index] *= MN_inv;

            // Calculate the flat index for the current (row, column) position in the 2D slice
            mn_index++;
            L_index = MN;
        }
    }

    return;
}

void SumAxis0_InPlace_Normalize_FFTW_Complex(int L, int M, int N, fftw_complex* complex_data){
  
    const int MN = (M * N);    // MN premultiplied
    int L_index = MN;    // Loop Index for axis 0
    // double MN_inv = 1.0;   // The normalization factor to scale the summed elements by the size of each 2D slice
    // Loop over each row (M) of the 2D slice
    #pragma omp parallel for
    for (int i = 0; i < MN; i++) {
        for (int k = 1; k < L; k++) {         
            // Add the element from the current layer (k) to the base layer (k = 0)
            complex_data[i][0] += complex_data[i + k*MN][0];
            complex_data[i][1] += complex_data[i + k*MN][1];
        }

        // Normalize the summed value by dividing by the size of the 2D slice (M * N)
        // complex_data[mn_index][0] *= MN_inv;
        // complex_data[mn_index][1] *= MN_inv;

    }
    return;
}

void ElementWiseAdd_InPlace_Double(int N, double* VolMod, double* VolAdd){

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        VolMod[i] = VolMod[i]+VolAdd[i];
    }

    return;
}

void ElementWiseDifference_InPlace_Double(int N, double* fp, double* img){

    #pragma omp parallel for
    for (size_t i = 0; i< N; i++){
        fp[i] = img[i] - fp[i];

    }

}

void ElementWiseLog2_InPlace_Double(int N, double* array, double offset){

    #pragma omp parallel for
    for (size_t i = 0; i< N; i++){
        if (array[i] <= 2.33e-10) {
            array[i] = -32;
        } else {
            array[i] = std::log2(array[i]) + offset;
        }
    }
}

void ElementWiseExp2_InPlace_Double(int N, double* array, double offset){

    #pragma omp parallel for
    for (size_t i = 0; i< N; i++){
        array[i] = std::exp2(array[i]*offset);
    }
}

void ElementWiseMult_InPlace_Double(int M, int N, int L, double* VolMod, double* VolMult){

    int TotalSize = M*N*L;
    // double norm = 1.0 / (M*(N/2 + 1));
    #pragma omp parallel for
    for (int i = 0; i < TotalSize; i++){
        VolMod[i] = VolMod[i]*VolMult[i];
    }

    return;
}

// Function: ElementWiseDivide_InPlace_Double
// Description: This function performs an element-wise division of two 2D arrays represented as 1D arrays 
//              in row-major order. The result is stored in the first array (`FP`), modifying it in place.
// 
// Parameters:
// - M: The number of rows in the 2D arrays.
// - N: The number of columns in the 2D arrays.
// - FP: Pointer to the 1D array of doubles representing the first 2D array (divisor), modified in place.
// - IMG: Pointer to the 1D array of doubles representing the second 2D array (dividend).
void ElementWiseDivide_InPlace_Double(int M, int N, double* fp, double* img) {
    int TotalSize = M * N;
    const double PAD = 1e-9; // Small value to avoid division by zero

    // Parallelize the loop using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < TotalSize; i++) {
        fp[i] = img[i] / (fp[i] + PAD);
    }
}

// Function: ElementWiseMult_InPlace_FFTW_Complex
// Description: This function performs an element-wise multiplication of two 3D arrays of complex numbers 
//              represented as 1D arrays of `fftw_complex` in row-major order. The result is stored 
//              in the first array (`VolWrite`), modifying it in place.
// 
// Parameters:
// - L: The size of the first dimension (depth or number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - VolWrite: Pointer to the 1D array of `fftw_complex` representing the first 3D array (modified in place).
// - VolRead: Pointer to the 1D array of `fftw_complex` representing the second 3D array (multiplier).
void ElementWiseMult_InPlace_FFTW_Complex(int TotalSize, fftw_complex* VolWrite, fftw_complex* VolRead) {

    double RealPartTemp;   // Temporary storage for the real part of the product

    #pragma omp parallel for private(RealPartTemp)
    for (int i = 0; i < TotalSize; i++) {
        // Calculate the flat index for the current (layer, row, column) position
        // index = k * M * N + j * N + i;        
        
        // Compute the real part of the product of the two complex numbers
        // RealPartTemp = (VolWrite.real * VolRead.real) - (VolWrite.imag * VolRead.imag)
        RealPartTemp = VolWrite[i][0] * VolRead[i][0] - VolWrite[i][1] * VolRead[i][1];
        
        // Compute the imaginary part of the product and store it in VolWrite.imag
        // VolWrite.imag = (VolWrite.real * VolRead.imag) + (VolWrite.imag * VolRead.real)
        VolWrite[i][1] = VolWrite[i][0] * VolRead[i][1] + VolWrite[i][1] * VolRead[i][0];
        
        // Update the real part of VolWrite with the previously computed RealPartTemp
        VolWrite[i][0] = RealPartTemp;
    }
    return;
}

void ElementwiseMultConjugate_FFTW_Complex(int L, int M, int N, 
                                           fftw_complex* dest, 
                                           fftw_complex* VolRatio_FT, 
                                           fftw_complex* PSF_FT) {
    int MN = M * N;  // Precompute M * N
    double RealPartTemp;

    #pragma omp parallel for private(RealPartTemp)
    for (int k = 0; k < L; k++) {
        for (int ij = 0; ij < MN; ij++) {
            int index = k * MN + ij;  // Calculate flat index for 3D array

            // Perform element-wise multiplication and conjugation
            RealPartTemp = VolRatio_FT[ij][0] * PSF_FT[index][0] + 
                                  VolRatio_FT[ij][1] * PSF_FT[index][1];
            dest[index][1] = -VolRatio_FT[ij][0] * PSF_FT[index][1] + 
                              VolRatio_FT[ij][1] * PSF_FT[index][0];
            dest[index][0] = RealPartTemp;
        }
    }
    return;
}


void SumAxis0_InPlace_Normalize_FFTW_ComplexF(int L, int M, int N, fftwf_complex* complex_data){
  
    const int MN = (M * N);    // MN premultiplied
    int L_index = MN;    // Loop Index for axis 0
    // double MN_inv = 1.0;   // The normalization factor to scale the summed elements by the size of each 2D slice
    // Loop over each row (M) of the 2D slice
    #pragma omp parallel for
    for (int i = 0; i < MN; i++) {
        for (int k = 1; k < L; k++) {         
            // Add the element from the current layer (k) to the base layer (k = 0)
            complex_data[i][0] += complex_data[i + k*MN][0];
            complex_data[i][1] += complex_data[i + k*MN][1];
        }

        // Normalize the summed value by dividing by the size of the 2D slice (M * N)
        // complex_data[mn_index][0] *= MN_inv;
        // complex_data[mn_index][1] *= MN_inv;

    }
    return;
}

void ElementWiseAdd_InPlace_Float(int N, float* VolMod, float* VolAdd){

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        VolMod[i] = VolMod[i] + VolAdd[i];
    }

    return;
}

void ElementWiseDifference_InPlace_Float(int N, float* fp, float* img){

    #pragma omp parallel for
    for (size_t i = 0; i< N; i++){
        fp[i] = img[i] - fp[i];
    }
}

void ElementWiseLog2_InPlace_Float(int N, float* array, float offset){

    #pragma omp parallel for
    for (size_t i = 0; i< N; i++){
        if (array[i] <= 2.33e-10) {
            array[i] = -32;
        } else {
            array[i] = std::log2f(array[i]) + offset;
        }
    }
}

void ElementWiseExp2_InPlace_Float(int N, float* array, float offset){

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        array[i] = std::exp2f(array[i]*offset);
    }
    
}


void ElementWiseMult_InPlace_Float(int M, int N, int L, float* VolMod, float* VolMult){

    int TotalSize = M*N*L;
    // double norm = 1.0 / (M*(N/2 + 1));
    #pragma omp parallel for
    for (int i = 0; i < TotalSize; i++){
        VolMod[i] = VolMod[i]*VolMult[i];
    }

    return;
}

// Function: ElementWiseDivide_InPlace_Double
// Description: This function performs an element-wise division of two 2D arrays represented as 1D arrays 
//              in row-major order. The result is stored in the first array (`FP`), modifying it in place.
// 
// Parameters:
// - M: The number of rows in the 2D arrays.
// - N: The number of columns in the 2D arrays.
// - FP: Pointer to the 1D array of doubles representing the first 2D array (divisor), modified in place.
// - IMG: Pointer to the 1D array of doubles representing the second 2D array (dividend).
void ElementWiseDivide_InPlace_Float(int M, int N, float* fp, float* img) {
    int TotalSize = M * N;
    const double PAD = 1e-9; // Small value to avoid division by zero

    // Parallelize the loop using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < TotalSize; i++) {
        fp[i] = img[i] / (fp[i] + PAD);
    }
}

// Function: ElementWiseMult_InPlace_FFTW_Complex
// Description: This function performs an element-wise multiplication of two 3D arrays of complex numbers 
//              represented as 1D arrays of `fftw_complex` in row-major order. The result is stored 
//              in the first array (`VolWrite`), modifying it in place.
// 
// Parameters:
// - L: The size of the first dimension (depth or number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - VolWrite: Pointer to the 1D array of `fftw_complex` representing the first 3D array (modified in place).
// - VolRead: Pointer to the 1D array of `fftw_complex` representing the second 3D array (multiplier).
void ElementWiseMult_InPlace_FFTW_ComplexF(int TotalSize, fftwf_complex* VolWrite, fftwf_complex* VolRead) {

    double RealPartTemp;   // Temporary storage for the real part of the product

    #pragma omp parallel for private(RealPartTemp)
    for (int i = 0; i < TotalSize; i++) {
        // Calculate the flat index for the current (layer, row, column) position
        // index = k * M * N + j * N + i;        
        
        // Compute the real part of the product of the two complex numbers
        // RealPartTemp = (VolWrite.real * VolRead.real) - (VolWrite.imag * VolRead.imag)
        RealPartTemp = VolWrite[i][0] * VolRead[i][0] - VolWrite[i][1] * VolRead[i][1];
        
        // Compute the imaginary part of the product and store it in VolWrite.imag
        // VolWrite.imag = (VolWrite.real * VolRead.imag) + (VolWrite.imag * VolRead.real)
        VolWrite[i][1] = VolWrite[i][0] * VolRead[i][1] + VolWrite[i][1] * VolRead[i][0];
        
        // Update the real part of VolWrite with the previously computed RealPartTemp
        VolWrite[i][0] = RealPartTemp;
    }
    return;
}

void ElementwiseMultConjugate_FFTW_ComplexF(int L, int M, int N, 
                                           fftwf_complex* dest, 
                                           fftwf_complex* VolRatio_FT, 
                                           fftwf_complex* PSF_FT) {
    int MN = M * N;  // Precompute M * N
    double RealPartTemp;

    #pragma omp parallel for private(RealPartTemp)
    for (int k = 0; k < L; k++) {
        for (int ij = 0; ij < MN; ij++) {
            int index = k * MN + ij;  // Calculate flat index for 3D array

            // Perform element-wise multiplication and conjugation
            RealPartTemp = VolRatio_FT[ij][0] * PSF_FT[index][0] + 
                                  VolRatio_FT[ij][1] * PSF_FT[index][1];
            dest[index][1] = -VolRatio_FT[ij][0] * PSF_FT[index][1] + 
                              VolRatio_FT[ij][1] * PSF_FT[index][0];
            dest[index][0] = RealPartTemp;
        }
    }
    return;
}
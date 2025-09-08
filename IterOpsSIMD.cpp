#include <immintrin.h> // Include the AVX2 intrinsics header
#include <fftw3.h>
#include <omp.h>

// SIMD optimized function
void SumAxis0_InPlace_FFTW_Complex_SIMD(int L, int M, int N, fftw_complex* complex_data) {
    int MN = M * N;       // MN pre-multiplied
    int L_index = MN;     // Loop Index for axis 0
    // double MN_inv = 1.0 / MN;  // Normalization factor

    // AVX register to hold the normalization factor
    // __m256d norm_factor = _mm256_set1_pd(MN_inv);

    #pragma omp parallel for
    for (int j = 0; j < M; j++) {
        for (int i = 0; i < N; i+=2) {
            int mn_index = j * N + i;

            // Load the initial value of the base layer (k = 0)
            __m256d sum = _mm256_load_pd(reinterpret_cast<double*>(&complex_data[mn_index]));

            for (int k = 1; k < L; k++) {
                // Load the current layer (k)
                __m256d current = _mm256_load_pd(reinterpret_cast<double*>(&complex_data[mn_index + k * MN]));

                // Add the current layer to the sum
                sum = _mm256_add_pd(sum, current);
            }

            // Normalize the summed values
            // sum = _mm256_mul_pd(sum, norm_factor);

            // Store the result back to the base layer (k = 0)
            _mm256_store_pd(reinterpret_cast<double*>(&complex_data[mn_index]), sum);
        }
    }
}

void ElementWiseMult_InPlace_Double_SIMD(int M, int N, int L, double* VolMod, double* VolMult) {
    int TotalSize = M * N * L;
    int simdSize = 4; // AVX processes 4 doubles per register (256 bits)

    // Process elements using SIMD
    int i = 0;
    for (; i <= TotalSize - simdSize; i += simdSize) {
        // Load 4 double elements from each array into AVX registers
        __m256d volModVec = _mm256_load_pd(&VolMod[i]);
        __m256d volMultVec = _mm256_load_pd(&VolMult[i]);

        // Perform element-wise multiplication
        __m256d resultVec = _mm256_mul_pd(volModVec, volMultVec);

        // Store the result back into VolMod
        _mm256_store_pd(&VolMod[i], resultVec);
    }

    // Process any remaining elements that don't fit in the SIMD blocks
    for (; i < TotalSize; i++) {
        VolMod[i] = VolMod[i] * VolMult[i];
    }
}

void ElementWiseMult_InPlace_FFTW_Complex_SIMD(int L, int M, int N, fftw_complex* VolWrite, fftw_complex* VolRead) {
    int index;
    #pragma omp parallel for
    for (int k = 0; k < L; k++) {
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < N; i += 4) { // Process 4 complex numbers at a time
                index = k * M * N + j * N + i;

                // Load complex numbers into SIMD registers
                __m256d vol_write_real = _mm256_loadu_pd(&VolWrite[index][0]);
                __m256d vol_write_imag = _mm256_loadu_pd(&VolWrite[index][1]);
                __m256d vol_read_real = _mm256_loadu_pd(&VolRead[index][0]);
                __m256d vol_read_imag = _mm256_loadu_pd(&VolRead[index][1]);

                // Perform complex multiplication
                __m256d real_part = _mm256_sub_pd(_mm256_mul_pd(vol_write_real, vol_read_real), _mm256_mul_pd(vol_write_imag, vol_read_imag));
                __m256d imag_part = _mm256_add_pd(_mm256_mul_pd(vol_write_real, vol_read_imag), _mm256_mul_pd(vol_write_imag, vol_read_real));

                // Store the results back
                _mm256_storeu_pd(&VolWrite[index][0], real_part);
                _mm256_storeu_pd(&VolWrite[index][1], imag_part);
            }
        }
    }
}

void ElementWiseDivide_InPlace_Double_SIMD(int M, int N, double* fp, double* img) {
    int TotalSize = M * N;
    int simdSize = 4; // AVX processes 4 doubles per register (256 bits)
    const double PAD = 1e-8;

    // Load padding value into a SIMD register
    __m256d padVec = _mm256_set1_pd(PAD);

    // Process elements using SIMD
    int i = 0;
    for (; i <= TotalSize - simdSize; i += simdSize) {
        // Load 4 double elements from each array into AVX registers
        __m256d fpVec = _mm256_load_pd(&fp[i]);
        __m256d imgVec = _mm256_load_pd(&img[i]);

        // Add PAD to fp elements to avoid division by zero
        __m256d fpPlusPad = _mm256_add_pd(fpVec, padVec);

        // Perform element-wise division: imgVec / fpPlusPad
        __m256d resultVec = _mm256_div_pd(imgVec, fpPlusPad);

        // Store the result back into fp
        _mm256_store_pd(&fp[i], resultVec);
    }

    // Process any remaining elements that don't fit in the SIMD blocks
    for (; i < TotalSize; i++) {
        fp[i] = img[i] / (fp[i] + PAD);
    }
}
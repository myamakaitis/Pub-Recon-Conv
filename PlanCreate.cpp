#include <fftw3.h>
#include <iostream>


fftw_plan create_batched_2d_fft_plan(int L, int M, int N, double* real_data, fftw_complex* complex_data) {

    // Stride and dimension setup for R2C transforms
    int rank = 2;                           // 2D FFTs
    int n[] = {M, N};                       // Dimensions of each 2D transform
    int nc[] = {M, (N/2)+1};
    int howmany = L;                        // Number of 2D FFTs to perform
    int idist = M * N;                      // Distance between the start of each 2D FFT in the real input
    int odist = M * (N / 2 + 1);            // Distance between each 2D FFT in the complex output
    int istride = 1;                        // Stride between elements in the same 2D plane (real input)
    int ostride = 1;                        // Output stride (complex)

    // Create the FFTW plan for batched 2D R2C FFTs
    fftw_plan plan = fftw_plan_many_dft_r2c(rank, n, howmany,
                                            real_data, n, istride, idist,
                                            complex_data, nc, ostride, odist,
                                            FFTW_MEASURE);

    return plan;
}

fftw_plan create_batched_2d_ifft_plan(int L, int M, int N, fftw_complex* complex_data, double* real_data) {

    // Stride and dimension setup for R2C transforms
    int rank = 2;                           // 2D FFTs
    int n[] = {M, N};                       // Dimensions of each 2D transform
    int nc[] = {M, (N/2)+1};
    int howmany = L;                        // Number of 2D FFTs to perform
    int idist = M * (N/2 + 1);              // Distance between the start of each 2D FFT in the real input
    int odist = M * N;                      // Distance between each 2D FFT in the complex output
    int istride = 1;                        // Stride between elements in the same 2D plane (real input)
    int ostride = 1;                        // Output stride (complex)

    // Create the FFTW plan for batched 2D R2C FFTs
    fftw_plan plan_inv = fftw_plan_many_dft_c2r(rank, n, howmany,
                                            complex_data, nc, istride, idist,
                                            real_data, n, ostride, odist,
                                            FFTW_MEASURE);

    return plan_inv;
}


fftwf_plan create_batched_2d_fft_plan_float(int L, int M, int N, float* real_data, fftwf_complex* complex_data) {

    // Stride and dimension setup for R2C transforms
    int rank = 2;                           // 2D FFTs
    int n[] = {M, N};                       // Dimensions of each 2D transform
    int nc[] = {M, (N/2)+1};
    int howmany = L;                        // Number of 2D FFTs to perform
    int idist = M * N;                      // Distance between the start of each 2D FFT in the real input
    int odist = M * (N / 2 + 1);            // Distance between each 2D FFT in the complex output
    int istride = 1;                        // Stride between elements in the same 2D plane (real input)
    int ostride = 1;                        // Output stride (complex)

    // Create the FFTW plan for batched 2D R2C FFTs
    fftwf_plan plan = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                            real_data, n, istride, idist,
                                            complex_data, nc, ostride, odist,
                                            FFTW_MEASURE);

    return plan;
}

fftwf_plan create_batched_2d_ifft_plan_float(int L, int M, int N, fftwf_complex* complex_data, float* real_data) {

    // Stride and dimension setup for R2C transforms
    int rank = 2;                           // 2D FFTs
    int n[] = {M, N};                       // Dimensions of each 2D transform
    int nc[] = {M, (N/2)+1};
    int howmany = L;                        // Number of 2D FFTs to perform
    int idist = M * (N/2 + 1);              // Distance between the start of each 2D FFT in the real input
    int odist = M * N;                      // Distance between each 2D FFT in the complex output
    int istride = 1;                        // Stride between elements in the same 2D plane (real input)
    int ostride = 1;                        // Output stride (complex)

    // Create the FFTW plan for batched 2D R2C FFTs
    fftwf_plan plan_inv = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                            complex_data, nc, istride, idist,
                                            real_data, n, ostride, odist,
                                            FFTW_MEASURE);

    return plan_inv;
}
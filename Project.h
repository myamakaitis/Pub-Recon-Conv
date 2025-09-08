#ifndef PROJECT_H
#define PROJECT_H

#include <fftw3.h>

// Function: ProjForward
// Description: Performs the forward projection of a volume using FFT and a PSF, 
//              followed by normalization and an inverse FFT.
// Parameters:
// - L: The size of the first dimension (number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - PlanForwardBatch: FFTW plan for executing the forward FFT.
// - vol_ft: Pointer to the FFTW complex array representing the volume in Fourier space.
// - psf_ft: Pointer to the FFTW complex array representing the PSF in Fourier space.
// - PlanBackward: FFTW plan for executing the inverse FFT.
void ProjForward(int L, int M, int N, 
                 fftw_plan PlanForwardBatch, 
                 fftw_complex* vol_ft, 
                 fftw_complex* psf_ft,
                 fftw_plan PlanBackward);

// Function: ProjBackward
// Description: Performs the backward projection using FFT, applying the PSF conjugate multiplication
//              and summing along the first axis.
// Parameters:
// - L: The size of the first dimension (number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - PlanForward: FFTW plan for executing the forward FFT.
// - img_fp_ft: Pointer to the FFTW complex array representing the image Fourier projection.
// - psf_ft: Pointer to the FFTW complex array representing the PSF in Fourier space.
// - vol_ft: Pointer to the FFTW complex array where results will be stored.
// - PlanBackwardBatch: FFTW plan for executing the inverse FFT.
void ProjBackward(int L, int M, int N,
                  fftw_plan PlanForward,
                  fftw_complex* img_fp_ft,
                  fftw_complex* psf_ft,
                  fftw_complex* vol_ft,
                  fftw_plan PlanBackwardBatch);

// Function: ProjForward
// Description: Performs the forward projection of a volume using FFT and a PSF, 
//              followed by normalization and an inverse FFT.
// Parameters:
// - L: The size of the first dimension (number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - PlanForwardBatch: FFTW plan for executing the forward FFT.
// - vol_ft: Pointer to the FFTW complex array representing the volume in Fourier space.
// - psf_ft: Pointer to the FFTW complex array representing the PSF in Fourier space.
// - PlanBackward: FFTW plan for executing the inverse FFT.
void ProjForward_Float(int L, int M, int N, 
                 fftwf_plan PlanForwardBatch, 
                 fftwf_complex* vol_ft, 
                 fftwf_complex* psf_ft,
                 fftwf_plan PlanBackward);

// Function: ProjBackward
// Description: Performs the backward projection using FFT, applying the PSF conjugate multiplication
//              and summing along the first axis.
// Parameters:
// - L: The size of the first dimension (number of layers).
// - M: The size of the second dimension (number of rows).
// - N: The size of the third dimension (number of columns).
// - PlanForward: FFTW plan for executing the forward FFT.
// - img_fp_ft: Pointer to the FFTW complex array representing the image Fourier projection.
// - psf_ft: Pointer to the FFTW complex array representing the PSF in Fourier space.
// - vol_ft: Pointer to the FFTW complex array where results will be stored.
// - PlanBackwardBatch: FFTW plan for executing the inverse FFT.
void ProjBackward_Float(int L, int M, int N,
                  fftwf_plan PlanForward,
                  fftwf_complex* img_fp_ft,
                  fftwf_complex* psf_ft,
                  fftwf_complex* vol_ft,
                  fftwf_plan PlanBackwardBatch);

#endif // PROJECT_H
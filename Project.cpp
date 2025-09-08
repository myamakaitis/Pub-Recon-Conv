#include <fftw3.h>
#include <iostream> 
#include "IterOps.h"

void ProjForward(int L, int M, int N, 
                fftw_plan PlanForwardBatch, 
                fftw_complex* vol_ft, 
                fftw_complex* psf_ft,
                fftw_plan PlanBackward
                )
{
    // execute plan
    fftw_execute(PlanForwardBatch);

    // Elementwise Multiplication
    ElementWiseMult_InPlace_FFTW_Complex(L*M*(N/2 + 1), vol_ft, psf_ft);

    //
    SumAxis0_InPlace_Normalize_FFTW_Complex(L, M, (N/2 + 1), vol_ft);

    // execute Inverse FFT to FP
    fftw_execute(PlanBackward);
        
    return;
}

void ProjBackward(int L, int M, int N,
                    fftw_plan PlanForward,
                    fftw_complex* img_fp_ft,
                    fftw_complex* psf_ft,
                    fftw_complex* vol_ft,
                    fftw_plan PlanBackwardBatch){

    // execute plan on IMG / Forward Projection Ratio
    fftw_execute(PlanForward);

    // Multiply The Fourier transform of the IMG / Forward Projection Ratio by each layer of the the conjugate of the PSF_FT
    ElementwiseMultConjugate_FFTW_Complex(L, M, (N/2 + 1), vol_ft, img_fp_ft, psf_ft);

    // Perform an inverse FFT on each layer of vol_ft
    fftw_execute(PlanBackwardBatch);    

    return;
}

void ProjForward_Float(int L, int M, int N, 
                fftwf_plan PlanForwardBatch, 
                fftwf_complex* vol_ft, 
                fftwf_complex* psf_ft,
                fftwf_plan PlanBackward
                )
{
    // execute plan
    fftwf_execute(PlanForwardBatch);

    // Elementwise Multiplication
    ElementWiseMult_InPlace_FFTW_ComplexF(L*M*(N/2 + 1), vol_ft, psf_ft);

    //
    SumAxis0_InPlace_Normalize_FFTW_ComplexF(L, M, (N/2 + 1), vol_ft);

    // execute Inverse FFT to FP
    fftwf_execute(PlanBackward);
        
    return;
}

void ProjBackward_Float(int L, int M, int N,
                    fftwf_plan PlanForward,
                    fftwf_complex* img_fp_ft,
                    fftwf_complex* psf_ft,
                    fftwf_complex* vol_ft,
                    fftwf_plan PlanBackwardBatch){

    // execute plan on IMG / Forward Projection Ratio
    fftwf_execute(PlanForward);

    // Multiply The Fourier transform of the IMG / Forward Projection Ratio by each layer of the the conjugate of the PSF_FT
    ElementwiseMultConjugate_FFTW_ComplexF(L, M, (N/2 + 1), vol_ft, img_fp_ft, psf_ft);

    // Perform an inverse FFT on each layer of vol_ft
    fftwf_execute(PlanBackwardBatch);    

    return;
}
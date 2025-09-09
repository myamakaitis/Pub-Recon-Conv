# Pure Python Implmentation of RLD

import numpy as np
from numpy.fft import rfft2, irfft2


# Projection operation with FT of PSF not precomputed
def ProjF_noFT(Vol, PSF):
    L, M, N = Vol.shape

    FP = np.zeros((M, N), Vol.dtype)

    for k in range(L):
        FP[:, :] += irfft2(rfft2(Vol[k, :, :]) * rfft2(PSF[k, :, :]))

    return FP

# Forward Projection w/ precomputed FT of PSF
def ProjF(Vol, PSF_FT, Vol_FT, FP):
    L, M, N = Vol.shape

    for k in range(L):
        # Convolution in frequency between each volume slice and corresponding PSF
        Vol_FT[k, :, :] = rfft2(Vol[k, :, :])*PSF_FT[k, :, :]
    
    # Forward Projection = Sum of convolutions
    # Using lineariy the summation is done in frequency domain
    FP[:, :] = irfft2(np.sum(Vol_FT, axis=0))

# Backward Projection
def ProjB(FP, FP_FT, PSF_FT, Vol_FT, Vol_Temp):
    L, M, N = Vol_Temp.shape

    FP_FT[:, :] = rfft2(FP)

    for k in range(L):
        # Cross-Correlation between the PSF at each plane and forward projection in frequency domain
        Vol_FT[k, :, :] = FP_FT*np.conjugate(PSF_FT[k, :, :])

        # Back to spatial domain        
        Vol_Temp[k, :, :] = irfft2(Vol_FT[k, :, :])

# RLD iterations
def RLD(
        Vol: np.ndarray,
        PSF: np.ndarray,
        Img: np.ndarray,
        Iters: int):
    
    L, M, N = Vol.shape
    
    # Temporary volume used during iterations
    Vol_Temp = np.zeros((L, M, N), Vol.dtype)

    # Array to hold the (2d) Fourier Transforms of the volume
    Vol_FT = np.zeros((L, M, (N//2 + 1)), dtype=np.complex128)
    
    # Array to hold the (2d) Fourier Transforms of the PSF stack
    PSF_FT = np.zeros((L, M, (N//2 + 1)), dtype=np.complex128)
    for k in range(L):
        PSF_FT[k, :, :] = rfft2(PSF[k, :, :])

    # Forward projection
    FP = np.zeros((M, N), Img.dtype)

    # Fourier transform of forward projection
    FP_FT = np.zeros((M, (N//2 + 1)), dtype=np.complex128)
    
    for _ in range(Iters):
        
        # Project the volume
        ProjF(Vol, PSF_FT, Vol_FT, FP)

        # Calculate the corrections
        FP[:, :] = Img/(FP+1e-12)

        # Back project corrections
        ProjB(FP, FP_FT, PSF_FT, Vol_FT, Vol_Temp)

        # apply corrections
        Vol *= Vol_Temp

    # Calculate final forward projection
    ProjF(Vol, PSF_FT, Vol_FT, FP)
    return FP

# Calculate the region of the image the reconstructed volume and PSF cover
def CreateMask(EmptyVol, PSF):
    L, M, N = EmptyVol.shape

    FP = np.zeros((M, N), EmptyVol.dtype)

    ProjF_noFT(EmptyVol, PSF, FP)

    return FP / L

if __name__ == "__main__":

    # This Test code creates a simple PSF and volume to test the RLD function
    # It compares the results of the pure Python RLD function to the C++ RLD

    Iters = 50

    from RLD_Bindings import RLD_cpp, np_empty_16byteAligned

    L, M, N = 4, 6, 8
    shape = (L, M, N)

    psf = np_empty_16byteAligned(shape, np.float64)
    psf[:] = 0

    for i in range(0, L):
        psf[i, 0, 0] = 0.25
        psf[i, 0, -(i+1)] = 0.25
        psf[i, -(i+1), 0] = 0.25
        psf[i, -(i+1), -(i+1)] = 0.25

    psf_FT = np.zeros((L, M, (N//2 + 1)), dtype=np.complex128)
    for k in range(L):
        psf_FT[k, :, :] = rfft2(psf[k, :, :])

    vol0 = np.zeros(shape, dtype=np.float64)

    vol0[:, 0, 0] = 1
    # vol0[1:-2, M//2 + 2, N//2 - 2] = 1
    # vol0[L//2, M//2, N//2] = 1

    vol = np_empty_16byteAligned(shape, np.float64)
    vol[:] = 1
    vol_FT = np.zeros((L, M, (N//2 + 1)), dtype=np.complex128)


    img0 = np_empty_16byteAligned((M, N), np.float64)
    img0[:] = 0
    ProjF(vol0, psf_FT, vol_FT, img0)

    import matplotlib.pyplot as plt

    print("img0")
    # print(np.around(img0,2))
    print("psf")
    # print(np.around(psf, 2))

    fig, ax = plt.subplots()

    ax.imshow(img0)

    fig.show()

    from time import perf_counter

    t0 = perf_counter()
    img = RLD(vol, psf, img0, Iters)
    tf = perf_counter()

    print(f"RLD Time: {tf-t0} sec")

    print("vol")
    print(np.around(vol, 2))

    fig, ax = plt.subplots()

    ax.set_title("Reprojection")
    ax.imshow(img)
    fig.show()

    vol[:] = 1

    t0 = perf_counter()
    RLD_cpp(vol, psf, img0, 50)
    tf = perf_counter()
    print(f"RLD Time: {tf-t0} sec")

    print("vol")
    print(vol)


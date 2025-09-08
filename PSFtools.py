# Imports
import numpy as np
from cv2 import imread
# import pandas as pd
from typing import Sequence
from Bindings_ArrayChecks import np_empty_16byteAligned


def MakeEvenSize(Array2d):
    if Array2d.shape[0] % 2 == 1:
        Array2d = Array2d[:-1, :]
    if Array2d.shape[1] % 2 == 1:
        Array2d = Array2d[:, :-1]

    return Array2d

# Normalize PSF so that it sums to 1
# I found that this needs to be done iteratively for large images
def NormalizePSF(psf, dtype=np.float64, iters=5, clip=0.0):
    psf = psf.astype(dtype)

    psf[psf < clip*psf.max()] = 0

    for _ in range(iters):
        psf /= np.sum(psf)
        psfc = psf / np.sum(psf)
        psf = (psf + psfc) / 2

    return psf

def ReadPSF(psf_filepath, clip=0.0):

    return NormalizePSF(imread(psf_filepath, -1), clip=clip)

def ReadPSFstack(psf_filepaths: Sequence[str], dtype=np.float64, shift=True, clip=0.0):

    psf0 = imread(psf_filepaths[0], -1)
    M, N = psf0.shape
    L = len(psf_filepaths)

    PSFstack = np_empty_16byteAligned((L, M, N), dtype)

    for l, psf_path in enumerate(psf_filepaths):
        PSFstack[l, :, :] = ReadPSF(psf_path, clip=clip)

    if shift:
        PSFstack= np.fft.fftshift(PSFstack, axes=(1,2))

    return PSFstack, (L, M, N)

def ReadRawImage(ImgName, dtype, invert=False):

    img = np.ascontiguousarray(imread(ImgName, -1).astype(dtype))

    if invert:
        img[:, :] = img[:, :].max() - img[:, :]

    return img
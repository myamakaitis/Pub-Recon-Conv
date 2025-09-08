# Imports
import numpy as np
from cv2 import imwrite, medianBlur, GaussianBlur

from time import perf_counter
from Bindings_RLD import RLD_cpp, RLDf_cpp
from Bindings_ArrayChecks import np_empty_16byteAligned
from PSFtools import ReadPSFstack, ReadRawImage

from RLD import ProjF_noFT
from os import mkdir

from datetime import datetime

#%% Log
now = datetime.now()
RunTime = now.strftime("%Y_%m_%d_%H.%M.%S")

bit_depth = 16

# %%
FLAG0 = "LABEL"

# %%

z_cal_step = 20 # [um] between PSF images
z0 = 0 # PSF# representing z = 0

# If the central PSF is not centered use these offsets
# They are the offset between the center of the image and the center of the central view

VerticalOffset = -50 # [pixels], Positive is up ^
HorizontalOffset = 50 # [pixels], Positive is right ->

# Assuming Circular Elemental View
# This is the radius of the elemental image in pixels
EI_radius = 220 # [pixels]

# Depth Range
z_min = -2000 # [um]
z_max = 500 # [um]

# Step Size
z_step = 20 # [um]

zPlanes = np.arange(z_min, z_max+z_step/2, z_step)

# Number of Iters
ITERS = 50

# ITERS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)

# %% Precision
dtype_choice = {"single":np.float32, "double":np.float64}
rld_choice = {"single": RLDf_cpp, "double": RLD_cpp}

Precision = "single" # either single or double

DataType = dtype_choice[Precision]
RLD = rld_choice[Precision]

# %% Load PSF
PSF_filepaths = [f"PATH_TO_PSF_IMAGES/{z:.0f}.tif" for z in zPlanes]
PSFstack, (L, M, N) = ReadPSFstack(PSF_filepaths, dtype=DataType, shift=True)
# if the PSFs are centered (They should be) the shift Parameter performs an FFT shift on them
# they become ucentered so that the reconstruction is

print(f"Stack Shape: ({L} x {M} x {N})")

# %% VOLUME SETUP
def FillVol(L, M, N, EmptyVol, Radius, OffsetH = 0, OffsetV=0):

    # EmptyVol = np.zeros((self.nz, self.ny, self.nx), dtype=dtype)
    EmptyVol[:, :, :] = 1.0

    xpx = np.arange(0, N) - N//2 - OffsetH
    ypx = np.arange(0, M) - M//2 + OffsetV

    Xpx, Ypx = np.meshgrid(xpx, ypx)
    # R2 = Xpx**2 + Ypx**2
    # ValidSpace = R2 < Radius**2

    ValidSpace = (np.abs(Xpx) < Radius) * (np.abs(Ypx) < Radius*2/3)

    for i in range(L):
        EmptyVol[i, :, :] *= (ValidSpace) #.astype(dtype))

VOL = np_empty_16byteAligned((L, M, N), DataType)
FillVol(L, M, N, VOL, EI_radius, OffsetH = HorizontalOffset, OffsetV = VerticalOffset)
ROI_j = (M//2 - EI_radius - VerticalOffset, M//2 + EI_radius + 1 - VerticalOffset)
ROI_i = (N//2 - EI_radius + HorizontalOffset, N//2 + EI_radius + 1 + HorizontalOffset)

#%% MAKE MASK
def CreateMask(EmptyVol, PSF):

    FP = ProjF_noFT(EmptyVol, PSF)

    return FP / L

Mask = CreateMask(VOL, PSFstack)
Mask = Mask > 0.25*Mask.max()

# SAVE FORMAT
format = "npy" # either npy or tif

#%% IMG LOAD
# Loops through all images to reconstruct
# Assumes they are organized as such
# {img_folder}{img_prefix}{img_number}{img_postfix}
# Then loops through img_nums
img_folder = "PATH_TO_IMAGE_FOLDER/"
img_pre = img_folder + "IMG PREFIX"
img_nums = np.arange(1, 1) ## CHANGE THIS TO THE NUMBER OF IMAGES
img_post = ".tif"

for n in img_nums:
    print(n)

    imgfile = f"{img_pre}{n:04d}{img_post}"

    IMG_RAW = ReadRawImage(imgfile, DataType, invert=True)

    IMG_RAW *= Mask

    # ANY IMAGE PREPROCESSING STEPS CAN BE ADDED HERE
        # e.g. blurring, median filtering, background subtraction, etc.

    IMG = IMG_RAW

    FillVol(L, M, N, VOL, EI_radius, OffsetH = HorizontalOffset, OffsetV = VerticalOffset)

    FLAG = FLAG0 + f""

    t0 = perf_counter()
    RLD(VOL, PSFstack, IMG, ITERS)
    tf = perf_counter()
    print(f"RLD w/ C++ Time: {tf-t0:.1f} sec")

    # OUTPUT CHECKS
    if VOL.min() < -1e-16:
        raise ValueError(f"RLD output is negative, minimum = {VOL.min()} at {VOL.argmin()}")
    if VOL.max() == 0:
        if PSFstack.max() == 0:
            raise ValueError("RLD output and PSF are all zero")
        raise ValueError("RLD output all zeros")
    if np.isnan(VOL).any():
        raise ValueError("RLD output has NaNs")
    
    # CROP TO ROI    
    VOL_crop = VOL[:, ROI_j[0]:ROI_j[1], ROI_i[0]:ROI_i[1]]

    # BEFORE SAVING NORMALIZE
    VOL_crop /= VOL_crop.max()
    VOL_crop *= (2**(16) - 1)

    # CONVOLUTED WAY TO CREATE A FOLDER FOR EACH IMAGE
    RawImgPathFolder = f"{'.'.join(imgfile.split('.')[:-1])}"

    try: mkdir(f"{RawImgPathFolder}")
    except FileExistsError: pass

    try: mkdir(f"{RawImgPathFolder}/zStack_{RunTime}_{FLAG}")
    except FileExistsError: pass

    np.savez(f"{RawImgPathFolder}/zStack_{RunTime}_{FLAG}/log.npz", Z=zPlanes, VerticalOffset=VerticalOffset, HorizontalOffset=HorizontalOffset,
            z_min = z_min, z_max = z_max, z_step=z_step, ITERS = ITERS, EI_radius=EI_radius)
    
    if format == "npy":
        np.save(f"{RawImgPathFolder}/zStack_{RunTime}_{FLAG}/VOL_crop.npy", VOL_crop)

    if format == "tif":
        # BEFORE SAVING NORMALIZE
        VOL_crop /= VOL_crop.max()
        VOL_crop *= (2**(16) - 1)

        for k in range(L):
            Slice = VOL_crop[k, :, :].astype(np.uint16)
            imwrite(f"{RawImgPathFolder}/zStack_{RunTime}_{FLAG}/z{k:03d}.tif", Slice)

# %%

import ctypes
import numpy as np
import os

from Bindings_ArrayChecks import *

# Load the shared library
if os.name == "nt":  # Windows
    dll_name = "Recon.dll"
    dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
    rld_lib = ctypes.CDLL(dllabspath, winmode=0)
elif os.name == "posix":  # Linux / macOS
    rld_lib = ctypes.CDLL("./Recon.so")  # Use .dylib for macOS
#
# # Load the shared library
# if os.name == "nt":  # Windows
#     rld_lib = ctypes.CDLL("./RLD.dll")
# elif os.name == "posix":  # Linux / macOS
#     rld_lib = ctypes.CDLL("./librld.so")  # Use .dylib for macOS

# Define argument types and return type for the RLD function
rld_lib.RLD.argtypes = [
    ctypes.c_int,  # L
    ctypes.c_int,  # M
    ctypes.c_int,  # N
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int   # ITERS
]
rld_lib.RLD.restype = None  # Function returns void

rld_lib.RLDf.argtypes = [
    ctypes.c_int,  # L
    ctypes.c_int,  # M
    ctypes.c_int,  # N
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int   # ITERS
]
rld_lib.RLDf.restype = None  # Function returns void


def RLD_cpp(Vol, PSF, IMG, ITERS):
    """
    Wrapper around the RLD function for type checking, validation, and alignment checking.

    Args:
        Vol (np.ndarray): LxMxN array of Volume Intensity Guess.
        PSF (np.ndarray): LxMxN array of PSFs.
        IMG (np.ndarray): MxN array of Observed Intensity.
        ITERS (int): Number of RLD iterations.

    Raises:
        ValueError: If the input arrays do not match expected shapes, types, or alignment.
        TypeError: If any of the input arguments are not of the correct type.
    """

    L, M, N = Vol.shape

    # Check types of L, M, N, and ITERS
    if not all(isinstance(x, int) for x in [L, M, N, ITERS]):
        raise TypeError("L, M, N, and ITERS must be integers.")

    # Validate Vol, PSF, and IMG arrays
    validate_array(Vol, (L, M, N), "Vol")
    validate_array(PSF, (L, M, N), "PSF")
    validate_array(IMG, (M, N), "IMG")

    Vol = Vol.ravel()
    PSF = PSF.ravel()
    IMG = IMG.ravel()
    
    Vol_ptr = Vol.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    PSF_ptr = PSF.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    IMG_ptr = IMG.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Call the C++ RLD function
    rld_lib.RLD(L, M, N, Vol_ptr, PSF_ptr, IMG_ptr, ITERS)

    # Example of post-processing or additional steps if needed
    print("RLD function executed successfully.")
    
    return Vol, PSF, IMG, ITERS


def RLDf_cpp(Vol, PSF, IMG, ITERS):
    """
    Wrapper around the RLD function for type checking, validation, and alignment checking.

    Args:
        Vol (np.ndarray): LxMxN array of Volume Intensity Guess.
        PSF (np.ndarray): LxMxN array of PSFs.
        IMG (np.ndarray): MxN array of Observed Intensity.
        ITERS (int): Number of RLD iterations.

    Raises:
        ValueError: If the input arrays do not match expected shapes, types, or alignment.
        TypeError: If any of the input arguments are not of the correct type.
    """

    L, M, N = Vol.shape

    # Check types of L, M, N, and ITERS
    if not all(isinstance(x, int) for x in [L, M, N, ITERS]):
        raise TypeError("L, M, N, and ITERS must be integers.")

    # Validate Vol, PSF, and IMG arrays
    validate_array(Vol, (L, M, N), "Vol", expected_dtype=np.float32)
    validate_array(PSF, (L, M, N), "PSF", expected_dtype=np.float32)
    validate_array(IMG, (M, N), "IMG", expected_dtype=np.float32)

    Vol = Vol.ravel()
    PSF = PSF.ravel()
    IMG = IMG.ravel()
    
    Vol_ptr = Vol.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    PSF_ptr = PSF.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    IMG_ptr = IMG.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C++ RLD function
    rld_lib.RLDf(L, M, N, Vol_ptr, PSF_ptr, IMG_ptr, ITERS)

    # Example of post-processing or additional steps if needed
    print("RLD function executed successfully.")
    
    return Vol, PSF, IMG, ITERS



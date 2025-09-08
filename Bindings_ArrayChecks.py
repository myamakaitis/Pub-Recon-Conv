import numpy as np


def np_empty_16byteAligned(shape, dtype):
    # https://stackoverflow.com/questions/9895787/memory-alignment-for-fast-fft-in-python-using-shared-arrays

    size = 1
    for dim in shape:
        size *= dim

    dtype = np.dtype(dtype)
    nbytes = size * dtype.itemsize
    buf = np.empty(nbytes + 16, dtype=np.uint8)
    start_index = -buf.ctypes.data % 16
    return buf[start_index:start_index + nbytes].view(dtype).reshape(shape)

def is_aligned_16(array):
    """
    Check if a numpy array is 16-byte aligned.

    Args:
        array (np.ndarray): The numpy array to check.

    Returns:
        bool: True if the array is 16-byte aligned, False otherwise.
    """
    address = array.__array_interface__['data'][0]
    return address % 16 == 0


def validate_array(array, expected_shape, array_name, expected_dtype=np.float64):
    """
    Validate that the given array meets the required conditions.

    Args:
        array (np.ndarray): The numpy array to validate.
        expected_shape (tuple): The expected shape of the array.
        array_name (str): Name of the array for error messages.

    Raises:
        ValueError: If the array does not meet the expected shape, type, or alignment.
    """
    # Check that the array is a numpy array of type float64 with the expected shape
    if not isinstance(array, np.ndarray) or array.dtype != expected_dtype or array.shape != expected_shape:
        raise ValueError(f"{array_name} must be a numpy array of shape {expected_shape} and type float64.")

    # Ensure that the array is C-contiguous in memory
    if not array.flags['C_CONTIGUOUS']:
        raise ValueError(f"{array_name} array must be C-contiguous in memory.")

    # Check if the array is 16-byte aligned
    if not is_aligned_16(array):
        raise ValueError(f"{array_name} array is not 16-byte aligned.")
    
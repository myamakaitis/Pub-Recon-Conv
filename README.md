Special implementations of Richardson-Lucy and SMART reconstructions 
for tomographic reconstruction for the cases when projection operations
can be modeled in terms of convolutions.

Algorithms themselves are implemented in C++

The C++ files are compimled to the libary Recon.dll

Python calls to these functions are in the associated  (Alg)_Bindings.py Files

A conda python evironment with all required libraries is specified in img-env.yml

These functions require special inputs, RLD_Batch.py and RLD_Experimental.ipynb are two examples of high level calls to the C++ implementations 

The required inputs are:
* List of Filepaths to Point spread functions (PSFs)
* Filepath to Raw Image (Must be same size as PSF images)
* Choice: Single / Double Precision
* Number of Iterations
* Optional Label

This Tool use the FFTW library. It is required to compile Recon.(dll/so) and to run
https://www.fftw.org/

All arguments that need to be passed to the C++ compiler are included in the "Build.txt" File
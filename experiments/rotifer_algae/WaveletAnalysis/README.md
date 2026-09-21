# WaveletAnalysis

Functions for time continuous wavelet transform and 
bivariate wavelet analysis, implemented in Julia.


**Description**: A collection of simple functions to perform time continunuous wavelet transform (CWT) and bivariate wavelet analysis, including wavelet cross spectrum (WCS), scale-time averaging, amplitude-phase separation, wavelet coherence (WCO), significance testing, and plotting. All functions assume that input data contain time series of real (or integer) numbers, given at a constant sampling rate (a function to handle missing data and to interpolate irregular data is provided).

Further included are some small function for fft-transform of real valued
signals, plus (naive) power spectrum and Hilbert transform. 

The code for the wavelet transform (in the function `src/wavelet.jl`) essentially is a straightforward Julia translation of the 1d Wavelet transform from [Torrence and Compo](http://paos.colorado.edu/research/wavelets/) (see original copyright below). 

The other functions are extensions for bivariate wavelet and phase analysis. 
This repository does not aspire to be exhaustive. 
The repository just contains a colloection of functions that I developed in the context of a recent paper and want to make open for public use:  
**Blasius et al. (2019) Long-term cyclic persistence in an experimental predator-prey system.
Nature. [DOI](https://doi.org/10.1038/s41586-019-1857-0)**  
*Please acknowledge the use of this software by citing the above publication*.

**Data**: The folder `data` contains some sample time series, e.g. the El Nino time series, the Canadian hare-lynx cycle, and the predator-prey time series from the reference Blasius et al (2019)

**Code**: The folder `src` contains the main algorithms, the folder `examples` contains some worked-out examples, demonstrating how to apply the algorithms and visualize the results.




**Original copyright for the wavelet transform code**:

> Copyright (C) 1995-2004, Christopher Torrence and Gilbert P. Compo
>
> This software may be used, copied, or redistributed as long as it is not
> sold and this copyright notice is reproduced on each copy made. This
> routine is provided as is without any express or implied warranties
> whatsoever.
>
> Notice: Please acknowledge the use of the above software in any publications:
>   Wavelet software was provided by C. Torrence and G. Compo,
>     and is available at URL: http://paos.colorado.edu/research/wavelets/.
>
> Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
>           Wavelet Analysis. _Bull. Amer. Meteor. Soc._, 79, 61-78.
>
> Please send a copy of such publications to either C. Torrence or G. Compo:
> Dr. Christopher Torrence               Dr. Gilbert P. Compo
> Research Systems, Inc.                 Climate Diagnostics Center
> 4990 Pearl East Circle                 325 Broadway R/CDC1
> Boulder, CO 80301, USA                 Boulder, CO 80305-3328, USA
> E-mail: chris[AT]rsinc[DOT]com         E-mail: compo[AT]colorado[DOT]edu




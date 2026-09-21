using FFTW, StatsFuns, Statistics, SpecialFunctions

# Some functions for continous wavelet transfrom and cross-spectra
#
#
# This file is based on a straightforward Julia translation of the
# code for 1d Wavelet transform from Torrence and Compo
#
# There is no intention to have a fancy, performant, or elegant "translation"
# it just should be able to "do stuff"
# The copyright remains by Torrence and Compo (see below)
# This code is provided without any warranties
# by Bernd Blasius (bernd.blasius@gmail.com)
#
#  HERE GOES THE ORIGINAL COPYRIGHT
#----------------------------------------------------------------------------
#   Copyright (C) 1995-2004, Christopher Torrence and Gilbert P. Compo
#
#   This software may be used, copied, or redistributed as long as it is not
#   sold and this copyright notice is reproduced on each copy made. This
#   routine is provided as is without any express or implied warranties
#   whatsoever.
#
# Notice: Please acknowledge the use of the above software in any publications:
#    ``Wavelet software was provided by C. Torrence and G. Compo,
#      and is available at URL: http://paos.colorado.edu/research/wavelets/''.
#
# Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
#            Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.
#
# Please send a copy of such publications to either C. Torrence or G. Compo:
#  Dr. Christopher Torrence               Dr. Gilbert P. Compo
#  Research Systems, Inc.                 Climate Diagnostics Center
#  4990 Pearl East Circle                 325 Broadway R/CDC1
#  Boulder, CO 80301, USA                 Boulder, CO 80305-3328, USA
#  E-mail: chris[AT]rsinc[DOT]com         E-mail: compo[AT]colorado[DOT]edu
#----------------------------------------------------------------------------



#WAVE_SIGNIF  Significance testing for the 1D Wavelet transform WAVELET
#
#   [signif,fft_theor] = ...
#      wave_signif(y,dt,scale,sigtest,lag1,siglvl,dof,mother,param)
#
# INPUTS:
#    y = the time series, or, the VARIANCE of the time series.
#        (If this is a single number, it is assumed to be the variance...)
#    dt = amount of time between each Y value, i.e. the sampling time.
#    scale = the vector of scale indices, from previous call to WAVELET.
#
# OUTPUTS:
#    signif = significance levels as a function of SCALE
#    fft_theor = output theoretical red-noise spectrum as fn of PERIOD
#
# OPTIONAL INPUTS:
# Note * setting any of the following to -1 will cause the default value to be used
#
#    sigtest = 0, 1, or 2.    If omitted, then assume 0.
#
#         If 0 (the default), then just do a regular chi-square test,
#             i.e. Eqn (18) from Torrence & Compo.
#         If 1, then do a "time-average" test, i.e. Eqn (23).
#             In this case, dof should be set to NA, the number
#             of local wavelet spectra that were averaged together.
#             For the Global Wavelet Spectrum, this would be NA=N,
#             where N is the number of points in your time series.
#         If 2, then do a "scale-average" test, i.e. Eqns (25)-(28).
#             In this case, dof should be set to a
#             two-element vector [S1,S2], which gives the scale
#             range that was averaged together.
#             e.g. if one scale-averaged scales between 2 and 8,
#             then dof=[2,8].
#
#    lag1 = LAG 1 Autocorrelation, used for signif levels. Default is 0.0
#
#    siglvl = significance level to use. Default is 0.95
#
#    dof = degrees-of-freedom for signif test.
#         IF sigtest=0, then (automatically) dof = 2 (or 1 for mother="DOG")
#         IF sigtest=1, then dof = na, the number of times averaged together.
#         IF sigtest=2, then dof = [s1,s2], the range of scales averaged.
#
#      Note: IF sigtest=1, then dof can be a vector (same length as scale's),
#            in which case na is assumed to vary with scale.
#            This allows one to average different numbers of times
#            together at different scales, or to take into account
#            things like the Cone of Influence.
#            See discussion following Eqn (23) in Torrence & Compo.
#
#----------------------------------------------------------------------------
#   Copyright (C) 1995-1998, Christopher Torrence and Gilbert P. Compo
#   University of Colorado, Program in Atmospheric and Oceanic Sciences.
#   This software may be used, copied, or redistributed as long as it is not
#   sold and this copyright notice is reproduced on each copy made.  This
#   routine is provided as is without any express or implied warranties
#   whatsoever.
#----------------------------------------------------------------------------
function wave_signif(y,dt,scale,sigtest=-1,lag1=-1,siglvl=-1,dof=-1,mother=-1,param=-1)
    n1 = length(y)
    j1 = length(scale) - 1
    #scale[1:j1+1] = scale1   # I don't think we need this
    s0 = minimum(scale)
    dj = log(scale[2]/scale[1])/log(2.)

    variance = n1 == 1 ? y : var(y)

    if sigtest == -1; sigtest = 0; end
    if lag1 == -1;    lag1 = 0.0; end
    if siglvl == -1;  siglvl = 0.95; end
    if mother == -1;  mother = "MORLET"; end

    mother = uppercase(mother)

    # get the appropriate parameters [see Table(2)]
    if mother == "MORLET"  #----------------------------------  Morlet
        if param == -1; param = 6.; end
        k0 = param
        fourier_factor = (4*pi)/(k0 + sqrt(2 + k0^2)) # Scale-->Fourier [Sec.3h]
        empir = [2.;-1;-1;-1]
        if k0 == 6; empir[2:4]=[0.776;2.32;0.60]; end
    elseif mother == "PAUL"  #--------------------------------  Paul
        if param == -1; param = 4.; end
        m = param
        fourier_factor = 4*pi/(2*m+1)
        empir = [2.,-1,-1,-1]
        if m == 4; empir[2:4]=[1.132;1.17;1.5]; end
    elseif mother == "DOG"  #---------------------------------  DOG
        if param == -1; param = 2.; end
        m = param
        fourier_factor = 2*pi*sqrt(2 ./ (2*m+1))
        empir = [1.;-1;-1;-1]
        if m == 2; empir[2:4] = [3.541;1.43;1.4]; end
        if m == 6; empir[2:4] = [1.966;1.37;0.97]; end
    else
        error("Mother must be one of MORLET,PAUL,DOG")
    end

    period = scale.*fourier_factor
    dofmin = empir[1]     # Degrees of freedom with no smoothing
    Cdelta = empir[2]     # reconstruction factor
    gamma_fac = empir[3]  # time-decorrelation factor
    dj0 = empir[4]        # scale-decorrelation factor

    freq = dt ./ period   # normalized frequency
    fft_theor = (1 .- lag1^2) ./ (1 .- 2*lag1*cos.(freq*2*pi) .+ lag1^2)  # [Eqn(16)]
    fft_theor = variance*fft_theor  # include time-series variance
    signif = fft_theor
    if dof == -1; dof = dofmin; end

    if sigtest == 0    # no smoothing, DOF=dofmin [Sec.4]
        dof = dofmin
        #chisquare = chisquare_inv(siglvl,dof)/dof # that's Torrence & Compo ..
        chisquare = chisqinvcdf(dof,siglvl)/dof    # .. in julia we use StatsFuns
        signif = fft_theor*chisquare   # [Eqn(18)]
    elseif sigtest == 1  # time-averaged significance
        if length(dof) == 1; dof=zeros(j1+1) .+ dof; end
        #truncate = find(dof .< 1)
        #dof[truncate] = ones(length(truncate))
        map!(z->max(1.0,z), dof, dof)
        dof = dofmin*sqrt.(1 .+ (dof*dt/gamma_fac ./ scale).^2 )   # [Eqn(23)]
        #truncate = find(dof .< dofmin)
        #dof[truncate] = dofmin*ones(length(truncate))   # minimum DOF is dofmin
        map!(z->max(dofmin,z), dof, dof)
        for a1 = 1:j1+1
            #chisquare = chisquare_inv(siglvl,dof[a1])/dof[a1]
            chisquare = chisqinvcdf(dof[a1],siglvl)/dof[a1]  # use StatsFuns
            signif[a1] = fft_theor[a1]*chisquare
        end
    elseif sigtest == 2  # time-averaged significance
        if length(dof) != 2
            error("DOF must be set to [S1,S2], the range of scale-averages")
        end
        if Cdelta == -1
            error("Cdelta & dj0 not defined for " * mother *
                " with param = " * string(param))
        end
        s1 = dof[1]
        s2 = dof[2]
        avg = find(z->((z>=s1) && (z<= s2)), scale)  # scales between S1 & S2
        navg = length(avg)
        if navg == 0
            error("No valid scales between " * string(s1) * " and " * string(s2))
        end
        Savg = 1 ./ sum(1 ./ scale[avg])     # [Eqn(25)]
        Smid = exp((log(s1)+log(s2))/2.)     # power-of-two midpoint
        dof = (dofmin*navg*Savg/Smid)*sqrt(1 + (navg*dj/dj0)^2)  # [Eqn(28)]
        fft_theor = Savg*sum(fft_theor[avg] ./ scale[avg])  # [Eqn(27)]
        #chisquare = chisquare_inv(siglvl,dof) ./ dof
        chisquare = chisqinvcdf.(dof,siglvl) ./ dof     # use StatsFuns
        signif = (dj*dt/Cdelta/Savg) .* fft_theor*chisquare    # [Eqn(26)]
    else
        error("sigtest must be either 0, 1, or 2")
    end

    signif, fft_theor
end

# **********************************************************************************
# **********************************************************************************

heavi(x) = x > 0. ? 1. : 0.    # Heaviside function

function wave_bases(mother,k,scale,param)
#   Computes the wavelet function as a function of Fourier frequency,
#   used for the wavelet transform in Fourier space.
#   (This program is called automatically by WAVELET)
#
# INPUTS:
#    mother = a string, equal to 'MORLET' or 'PAUL' or 'DOG'
#    k = a vector, the Fourier frequencies at which to calculate the wavelet
#    scale = a number, the wavelet scale
#    param = the nondimensional parameter for the wavelet function
#
# OUTPUTS:
#    daughter = a vector, the wavelet function
#    fourier_factor = the ratio of Fourier period to scale
#    coi = a number, the cone-of-influence size at the scale
#    dofmin = a number, degrees of freedom for each point in the wavelet power
#             (either 2 for Morlet and Paul, or 1 for the DOG)
#             (either 2 for Morlet and Paul, or 1 for the DOG)
#
#    Copyright statement of the original Matlab version
#----------------------------------------------------------------------------
#   Copyright (C) 1995-1998, Christopher Torrence and Gilbert P. Compo
#   University of Colorado, Program in Atmospheric and Oceanic Sciences.
#   This software may be used, copied, or redistributed as long as it is not
#   sold and this copyright notice is reproduced on each copy made.  This
#   routine is provided as is without any express or implied warranties
#   whatsoever.
#----------------------------------------------------------------------------
    mother = uppercase(mother)
    n = length(k)
    if mother == "MORLET"  #-----------------------------------  Morlet
       if param == -1; param = 6.; end
       k0 = param
       expnt = -(scale*k .- k0).^2 / 2 .* heavi.(k)
       norm = sqrt(scale*k[2])*(pi^(-0.25))*sqrt(n)      # total energy=N   [Eqn(7)]
       daughter = norm*exp.(expnt)
       daughter = daughter .* heavi.(k)                  # Heaviside step function
       fourier_factor = (4.0*pi)/(k0 + sqrt(2.0 + k0^2)) # Scale-->Fourier [Sec.3h]
       coi = fourier_factor/sqrt(2.0)                    # Cone-of-influence [Sec.3g]
       dofmin = 2                                        # Degrees of freedom
     elseif mother == "PAUL"  #-----------------------------------  Paul
       if param == -1; param = 4. ; end
       m = param
       expnt = -(scale.*k).* heavi.(k)
       norm = sqrt(scale*k[2])*(2^m/sqrt(m*prod(2:(2*m-1))))*sqrt(n)
       daughter = norm*((scale.*k).^m).*exp.(expnt)
       daughter = daughter.* heavi.(k)        # Heaviside step function
       fourier_factor = 4*pi/(2.0*m+1)
       coi = fourier_factor*sqrt(2.0)
       dofmin = 2
     elseif mother == "DOG"  #-----------------------------------  DOG
       if param == -1; param = 2.; end
       m = param
       expnt = -(scale.*k).^2 ./ 2.0
       norm = sqrt(scale*k[2]/gamma(m+0.5))*sqrt(n)
       daughter = -norm*(im^m)*((scale.*k).^m).*exp.(expnt)
       fourier_factor = 2*pi*sqrt(2 ./ (2*m+1))
       coi = fourier_factor/sqrt(2)
       dofmin = 1
     else
       error("Mother must be one of MORLET,PAUL,DOG")
     end
     daughter, fourier_factor, coi, dofmin
end


# **********************************************************************************
# **********************************************************************************

#   1D contunous Wavelet transform
#
#   Computes the wavelet transform of the vector Y (length N),
#   with sampling rate DT.
#
#   By default, the Morlet wavelet (k0=6) is used.
#   The wavelet basis is normalized to have total energy=1 at all scales.
#
#
# INPUTS:
#
#    y = the time series of length N.
#    dt = amount of time between each Y value, i.e. the sampling time.
#
# OUTPUTS:
#
#    wave is the WAVELET transform of y. This is a complex array
#    of dimensions (n,j1+1). real(wave) gives the WAVELET amplitude,
#    atan(imag(wave),real(wave))  OR angle(wave) gives the WAVELET phase.
#    The WAVELET power spectrum is abs2(wave).
#    Its units are sigma^2 (the time series variance).
#
#
# OPTIONAL INPUTS:
#
# *** Note *** setting any of the following to -1 will cause the default
#               value to be used.
#
#    pad = if set to 1 (default is 0), pad time series with enough zeroes to get
#         N up to the next higher power of 2. This prevents wraparound
#         from the end of the time series to the beginning, and also
#         speeds up the FFT's used to do the wavelet transform.
#         This will not eliminate all edge effects (see COI below).
#
#    dj = the spacing between discrete scales. Default is 0.25.
#         A smaller # will give better scale resolution, but be slower to plot.
#
#    s0 = the smallest scale of the wavelet.  Default is 2*DT.
#
#    j1 = the # of scales minus one. Scales range from s0 up to s0*2^(j1*dj),
#        to give a total of (j1+1) scales. Default is j1 = (log2(n dt/s0))/dj.
#
#    mother = the mother wavelet function.
#             The choices are 'MORLET', 'PAUL', or 'DOG'
#
#    param = the mother wavelet parameter.
#            For 'MORLET' this is k0 (wavenumber), default is 6.
#            For 'PAUL' this is m (order), default is 4.
#            For 'DOG' this is m (m-th derivative), default is 2.
#
#
# OUTPUTS:
#
#    period = the vector of "Fourier" periods (in time units) that corresponds
#           to the scaels.
#
#    scale = the vector of scale indices, given by s0*2^(j*dj), j=0...j1
#            where j1+1 is the total # of scales.
#
#    coi = if specified, then return the Cone-of-Influence, which is a vector
#        of N points that contains the maximum period of useful information
#        at that particular time.
#        Periods greater than this are subject to edge effects.
#        This can be used to plot coi lines on a contour plot by doing:
#
#             IN MATLAB:
#              contour(time,log(period),log(power))
#              plot(time,log(coi),'k')
#
#----------------------------------------------------------------------------
#   Copyright (C) 1995-2004, Christopher Torrence and Gilbert P. Compo
#
#   This software may be used, copied, or redistributed as long as it is not
#   sold and this copyright notice is reproduced on each copy made. This
#   routine is provided as is without any express or implied warranties
#   whatsoever.
#
# Notice: Please acknowledge the use of the above software in any publications:
#    ``Wavelet software was provided by C. Torrence and G. Compo,
#      and is available at URL: http://paos.colorado.edu/research/wavelets/''.
#
# Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
#            Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.
#
# Please send a copy of such publications to either C. Torrence or G. Compo:
#  Dr. Christopher Torrence               Dr. Gilbert P. Compo
#  Research Systems, Inc.                 Climate Diagnostics Center
#  4990 Pearl East Circle                 325 Broadway R/CDC1
#  Boulder, CO 80301, USA                 Boulder, CO 80305-3328, USA
#  E-mail: chris[AT]rsinc[DOT]com         E-mail: compo[AT]colorado[DOT]edu
#----------------------------------------------------------------------------

function wavelet(y,dt,pad=0,dj=-1,s0=-1,j1=-1,mother=-1,param=-1)

    n1 = length(y)
    if s0 == -1; s0=2*dt; end
    if dj == -1; dj = 1. / 4.; end
    if j1 == -1; j1=trunc(Int,(log(n1*dt/s0)/log(2))/dj); end
    if mother == -1; mother = "MORLET"; end

    #....construct time series to analyze, pad if necessary
    x = y .- mean(y)
    if pad == 1
        base2 = trunc(Int,log(n1)/log(2) + 0.4999)   # power of 2 nearest to N
        x = [x; zeros(2^(base2+1)-n1)]
    end
    n = length(x)

    #....construct wavenumber array used in transform [Eqn(5)]
    k = 1:trunc(Int,n/2)  
    k = k .* ((2. * pi)/(n*dt))
    k = [0.; k; -k[trunc(Int,(n-1)/2):-1:1]]
    #
    #....compute FFT of the (padded) time series
    f = fft(x)    # [Eqn(3)]

    #....construct SCALE array & empty PERIOD & WAVE arrays
    scale = s0*2 .^ ((0:j1)*dj)
    period = scale
    wave = zeros(Complex,j1+1,n)   # define the wavelet array

    # loop through all scales and compute transform
    fourier_factor = 0.0  # julia seems to need a default
    for a1 = 1:j1+1
       daughter,fourier_factor,coi,dofmin = wave_bases(mother,k,scale[a1],param)
       wave[a1,:] = ifft(f.*daughter)  # wavelet transform[Eqn(4)]
    end

    period = fourier_factor*scale
    #coi = coi*dt*[1e-5; collect(1:((n1+1)/2-1)); collect((n1/2-1):-1:1); 1e-5]  # COI [Sec.3g]
    coi = coi*dt*[1e-5; (1:((n1+1)/2-1)); ((n1/2-1):-1:1); 1e-5]  # COI [Sec.3g]
    wave = wave[:,1:n1]  # get rid of padding before returning

    wave, period, scale, coi
end

## Some auxilliary functions, used for continuous wavelet analysis
## Author Bernd Blasius (bernd.blasius@gmail.com)



function deal_missing(x,t)
  # remove missing data (in csv file this is coded as NaN)
  n = length(x)
  x1 = zeros(n)
  t1 = zeros(n)

  j=0
  for i = 1:n
    if !isnan(x[i])
       j += 1
       x1[j] = x[i]
       t1[j] = t[i]
    end
  end
  x1[1:j], t1[1:j]
end


function sample_regular(x,t,dt,keep_positive)
# sample time series x,t at regular sampling rate dt and interpolate missing values   
# needs julia package Dierckx

    # eliminate missing values, this yields new time vectors for every state
    xx,tx = deal_missing(x,t)

    # interpolation to a regular one-day sampling interval
    t_end = trunc(t[end])
    tt = 0:dt:t_end
    xr = evaluate(Spline1D(tx,xx),tt)   # regularly spaced time series

    # frequently we want to restrict time series to positive values
    # this may be violiated by the spline fitting which artificially may introduce 
    # negative values
    # this is a simple hack to remove this, but it works (at least in the case I was interested in)
    if keep_positive 
      xr = minimum(xr) < 0 ? xr .- minimum(xr) : xr
    end

    return tt,xr
end


function scramble(x,n)
# sample n random values from time series x
# to create surrogate data
    len = length(x)
    x[rand(1:len,n)]
end


function smooth(x,dt,nvoice,scale,sw=0.6)
# wavelet scale-time smoothing
# this function needs julia package DSP.jl
    xf = similar(x)  # smoothed data
    nscales, ntimes = size(x)
 
    # smooth in time, by filtering with a Gaussian
    for i=1:nscales
      s = scale[i]
      b = [exp(-0.5*(j*dt/s)^2)/(s*2*pi)  for j=1:ntimes]
      xf[i,:] = filtfilt(b,x[i,:])  
    end
 
    # smooth in scale according to Si et al (2008)
    # window sw=0.6 corresponds to decorrelation length of Morlet wavelet
    sw_tot = trunc(Int,sw*nvoice)
    b = (1/sw_tot)*ones(sw_tot)     # flat (rectangular) window -> moving average
    for i=1:ntimes 
      xf[:,i] = filtfilt(b,xf[:,i])
    end
    xf
 end

 function preprocess(x)
 # do some data preprocessing with a bandpass filter
    responsetype = Bandpass(1 ./ 12, 0.49; fs=1)
    designmethod = Butterworth(4)
    x = filt(digitalfilter(responsetype, designmethod), x)
    x
 end

 function preprocess1(t,x,dt)
    # bandpass filter in frequency domain (implemented by hand)
    sh = 500       #   sharpnes of filter function
    # minimum and maximum frequency cut-offs of the bandpass filter
    f1 = 1/16
    f2 = 1/5

    fmax = 1/(2*dt)           # highest possible frequency
    fmin = 1/(t[end] - t[1])  # lowest possible frequency
    freq = LinRange(fmin,2*fmax,length(t))
    # Fermi function
    fil_part1 = 1 ./ (exp.(sh*(freq .- f2)) .+ 1) .* 1. ./ (exp.(sh*(f1 .- freq)) .+ 1) 
    fil_part2 = fil_part1[length(fil_part1):-1:1]  # symmetrical counter part
    fil_freq = max.(fil_part1,fil_part2)           # merge the parts

    ft = fft(x)          # Fourier transformg
    phase = angle.(ft)   # phase
    ps = abs.(ft)        # power spectrum

    #   filter data
    ps_fil = ps .* fil_freq     # filter power spectra
    # recalculate fourier
    ft_fil = cos.(phase) .* ps_fil .+       #real part
              sin.(phase) .* ps_fil*im      #imaginary part
    fil = real.(ifft(ft_fil))            # retransformation
 end


# #########################################################
# some functions for circular Statistics

function unwrap(v,vmin,vmax)
  # unwrap phase angle (phase in multiples of pi)
    if v < vmin
        v += 2.0
    elseif v > vmax
        v -= 2.0
    end
  v
end

function circstats(ph)
    # ph is a vector if phases in units of 2 pi
    n = length(ph)

    c = sum(cos.(ph)) / n
    s = sum(sin.(ph)) / n

    R = sqrt(c*c + s*s)       # order parameter
    var = 1.0 - R             # cicurlar variance
    theta = atan(s,c)         # mean direction
    std = sqrt(-2. * log(R))  # circular standard deviation
    theta, var, std
end



# #########################################################
# function to indentify strongest co-oscillating components

function get_maxInd(wco, phs, n, ind1, ind2)
    # find maximal wavelet coherence at each time instance
    # search only within a range of indices (ind1, ind2) - corresponding to large power

    wco_max_ind = zeros(Int,n)   # vector of indices of maximal coherence
    wco_max  = zeros(n)          # vector of maximal
    angle_max = zeros(n)         # vector of correspondingphase differences
    for i=1:n
        ind = argmax(wco[ind1:ind2,i]) + ind1 - 1
        wco_max_ind[i] = ind
        wco_max[i]   = wco[ind,i]
        angle_max[i] = phs[ind,i] / pi   # we always measure phase in multiples of pi
    end
    angle_max = unwrap.(angle_max, -0.5, 1.5)   # restrict phases to range -0.5 .. 1.5

    wco_max_ind, angle_max, wco_max
end


function phasehist(angles, indices, delta, width)
    # calculate histogram of phase angles (given in multiples of pi)
    delta = 0.01
    angle_hist = fit(Histogram, angles[indices],closed=:right, -0.5:delta:1.5)
    h_angle = angle_hist.weights
    h_bins = pi* (angle_hist.edges[1][2:end])

    #smoothing of histogram
    filt = (1/width)*ones(width)   # filter window
    ll = length(h_angle)
    h_angle1 = [h_angle; h_angle ; h_angle]  # wrap the histogram to consider periodic boundaries
    h_angle1 = filtfilt(filt,h_angle1)
    h_angle = h_angle1[ll+1:2*ll]
    
    # normalization of histogram
    #h_angle = h_angle / sum(h_angle)    # this would ormalize the histogram
    h_angle = h_angle / maximum(h_angle) # here instead we want the max to be one

    h_angle, h_bins
end

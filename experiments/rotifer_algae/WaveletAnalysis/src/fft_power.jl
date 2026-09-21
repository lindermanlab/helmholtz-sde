##  Fast Fourier transform of a real-valued signal
##  + simple power spectrum
##  + Hilbert transform
##  Author Bernd Blasius (bernd.blasius@gmail.com)

# this files will need the Julia Package FFTW.jl

function power(x,fs)
    # simple power spectrum
    # x: time series
    # f: sampling frequency
    n = length(x)
    xbar = fftshift(fft(x))
    f = (-n/2:n/2-1)*(fs/n)    # zero-centered frequency range
    power= abs.(xbar).^2/n     # zero-centered power

    f, power
end

function power_real(x,fs)
    # simple power spectrum using rfft
    # x: time series
    # f: sampling frequency
    n = length(x)
    xbar = rfft(x)
    n1 = length(xbar)
    f = (0:n1-1) * (fs/n)     #frequency range
    power = abs.(xbar).^2/n    #power

    f, power
end

function hilbert(x,fs)
    # calculate the Hilbert transform of a real valued signal
    n = length(x)
    xbar = rfft(x)
    irfft(im*xbar,n) # Hilbert transform
end


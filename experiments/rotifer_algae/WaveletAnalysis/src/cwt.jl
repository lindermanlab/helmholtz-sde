##  Continuous wavelet analysis in julia
##  Author Bernd Blasius (bernd.blasius@gmail.com)

# for a more involved example, including plotting, wavelet power spectrum
# and significance levels, see test/cwt_test_elnino.jl 


using Statistics
include("wavelet.jl")


# continuoous wavelet transform (cwt)
function cwt(x,dt; 
             pad=1,        # pad the time series with zeros
             nvoices = 32, # total number of voices
             noctave = 4,  # total number of octaves
             s0 = 2*dt,    # starting scale
             mother = "MORLET",
             param = 6)
    # x : vector of time series
    # dt: sampling time interval
    # this function assumes that the time series is sampled equally
    # no checks are made to ensure this!!

    dj = 1 ./ nvoices
    j1 = noctave * nvoices
    
    # normalize time series, but store the variance
    variance = var(x)
    x = x .- mean(x)    
    x = x ./ std(x)

    wave, period, scale, coi = wavelet(x,dt,pad,dj,s0,j1,mother,param)

    # wavelet power spetrum
    power = variance * abs2.(wave)   

    # Global wavelet spectrum 
    global_ws = variance * mean(power,dims=2)   # time-average over all times

    return wave, period, power, global_ws, variance, scale, coi
end


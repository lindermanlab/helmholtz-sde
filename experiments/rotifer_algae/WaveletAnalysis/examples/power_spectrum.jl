##  Test of fft, power spectrum, and Hilbert transform
##  using as example of the ElNino time series
## 
##  Author Bernd Blasius (bernd.blasius@gmail.com)

module W

using CSV, PyPlot, FFTW, Statistics

include("../src/fft_power.jl")

function test_power_spectrum()
 # read in the data
  df = CSV.read("../data/ElNino_data.csv"; comment="#") 
  t   = df[:,1]  # time (year)
  sst = df[:,2]  # el nino signal (sea surface temperature SST)
  x   = df[:,3]  # el nino temperature anomaly

  fs = 1 ./ (mean(diff(t)))  # mean sampling frequency
  f, pow = power_real(x,fs)


  figure("El Nino Power")
  clf()
  subplot(2,1,1)
  plot(t,x)
  ylabel("El Nino")
  xlabel("time")

  subplot(2,1,2)
  plot(f, log10.(pow))
  xlabel("frequency")
  ylabel("power")

end

function test_hilbert()
  # read in the El Nino data
  df = CSV.read("../data/ElNino_data.csv"; comment="#") 
  t   = df[:,1]  # time (year)
  sst = df[:,2]  # el nino signal (sea surface temperature SST)
  x   = df[:,3]  # el nino temperature anomaly

  fs = 1 ./ (mean(diff(t)))  # mean sampling frequency
  h = hilbert(x,fs)

  figure("hilbert")
  clf()
  subplot(2,1,1)  # phase plane
  plot(t,x)  # El Nino
  plot(t,h,"r") # Hilber transform
  xlabel("time")
  ylabel("El Nino")

  subplot(2,1,2)  # phase plane
  plot(x,h)


end

#test_power_spectrum()
test_hilbert()

end # module



##  Test continuous wavelet transform and wavelet power spectrum
##  with the example of the ElNino time series
##  again closely following Torrence & Compo
## 
##  Author Bernd Blasius (bernd.blasius@gmail.com)

module W

using CSV, PyPlot

include("../src/cwt.jl")


function test_elnino()
  # read in the data
  df = CSV.read("../data/ElNino_data.csv"; comment="#") 
  t   = df[:,1]  # time (year)
  sst = df[:,2]  # el nino signal (sea surface temperature SST)
  x   = df[:,3]  # el nino temperature anomaly

  figure("elnino time series")
  clf()
  subplot(2,1,1)
  plot(t,sst)
  xlabel("time")
  ylabel("SST")
  subplot(2,1,2)
  plot(t,x)
  xlabel("time")
  ylabel("anomaly")


  dt = mean(diff(t))   # actually we would need perfectly regular sampling intervals
  nvoices = 16  # total number of voices
  noctave = 5   # total number of octaves
  s0 = 1.0      # starting scale at 1 years
  
  wave, period, power, global_ws, variance, scale, coi = 
    cwt(x,dt,noctave = noctave, nvoices= nvoices, s0=s0) 

  # Significance levels: (variance=1 for the normalized SST)
  lag1 = 0.72    # lag-1 autocorrelation for red noise background
  signif,fft_theor = wave_signif(variance,dt,scale,0,lag1)
  sig95 = signif .* ones(length(x))'  # expand signif --> (J+1)x(N) array
  sig95 = power ./ sig95              # where ratio > 1, power is significant

  # significance level for global spectrum
  dof = length(x) .- scale  # the -scale corrects for padding at edges
  global_signif,fft_theor = wave_signif(variance,dt,scale,1,lag1,-1,dof)

  # plotting scale range
  lminp = log2(minimum(period))
  lmaxp = log2(maximum(period))
  lrange = (trunc(lminp) :  trunc(lmaxp))


  figure("El Nino wavelet power spectrum")
  clf()
  # either use pcolormesh ..
    pcolormesh(t, log2.(period),power, cmap=PyPlot.cm.jet, alpha = 0.5)
  # ... or use contours
  levels = range(0, stop=maximum(power), length=200)
  contourf(t, log2.(period),power,levels, cmap=PyPlot.cm.jet )
  #
  plot(t,log2.(coi),"k")           # cone-of-influence, anything "below" is dubious
  contour(t, log2.(period),sig95, [-99,1], colors="black")   # 95% significance value
  ylabel("period length")
  xlabel("time")
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)

  figure("global spectrum")
  clf()
  plot(log2.(period),log2.(global_ws))
  plot(log2.(period),log2.(global_signif),"k.")  # 95% significance line
  xlabel("period")
  ylabel("log2(power)")
  xlim(lmaxp,lminp)
  xticks(lrange, 2 .^ lrange)
end

test_elnino()

end # module

##  Test continuous wavelet transform 
##  with classic hare-lynx data set
## 
##  Author Bernd Blasius (bernd.blasius@gmail.com)

module W

using CSV, DSP, PyPlot, StatsBase

include("../src/cwt.jl")
include("../src/tools.jl")
include("../src/wcs.jl")

function test_cwt_harelynx()
    # test continuous time wavelet transform (CWT) and
    # wavelet power spectrum

  # read in the data
  df = CSV.read("../data/Hare_Lynx.csv") 
  t   = df[:,1]  # time (year)
  h   = df[:,2]  # hare
  l   = df[:,3]  # lynx

  figure("Hare Lynx time series")
  clf()
  plot(t,h)
  plot(t,10*l)
  xlabel("time")
  ylabel("population abundances")

  # do the wavelet transform
  dt = 1    # sampling rate = 1 year
  s0 = 2.0  # starting scale = 1 year
  noctave = 4  # number of octaves
  nvoices = 100 # number of voices per octave

  wave1, period1, power1, global_ws1, variance1, scale1, coi1 = 
    cwt(h,dt,noctave = noctave, s0=s0, nvoices = nvoices) 
  wave2, period2, power2, global_ws2, variance2, scale2, coi2 = 
    cwt(l,dt,noctave = noctave, s0=s0, nvoices = nvoices) 

  
  # Significance levels: (variance=1 for the normalized SST)
  lag1 = 0.72    # lag-1 autocorrelation for red noise background
  signif1,fft_theor1 = wave_signif(variance1,dt,scale1,0,lag1)
  sig95_1 = signif1 .* ones(length(h))'  # expand signif --> (J+1)x(N) array
  sig95_1 = power1 ./ sig95_1   

  signif2,fft_theor2 = wave_signif(variance2,dt,scale2,0,lag1)
  sig95_2 = signif2 .* ones(length(l))'  # expand signif --> (J+1)x(N) array
  sig95_2 = power2 ./ sig95_2   

  # plotting scale range
  lminp = log2(minimum(period1))
  lmaxp = log2(maximum(period1))
  lrange = (trunc(lminp) :  trunc(lmaxp))

  figure("Hare-lynx wavelet power spectrum")
  clf()
  # plot wavelet power for hare
  subplot(2,1,1)
  # either use pcolormesh ..
     #pcolormesh(t, log2.(period1),power1, cmap=PyPlot.cm.jet, alpha = 0.5)
  # ... or use contours
  levels1 = range(0, stop=maximum(power1), length=200)
  contourf(t, log2.(period1),power1,levels1, cmap=PyPlot.cm.jet )
  plot(t,log2.(coi1),"k")           # cone-of-influence, anything "below" is dubious
  contour(t, log2.(period1),sig95_1, [-99,1], colors="black")   # 95% significance value
  ylabel("period length")
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)
   
  # plot wavelet power for lynx
  subplot(2,1,2)
  # either use pcolormesh ..
    #pcolormesh(t, log2.(period2),power2, cmap=PyPlot.cm.jet, alpha = 0.5)
  # ... or use contours
  levels2 = range(0, stop=maximum(power2), length=200)
  contourf(t, log2.(period2),power2,levels2, cmap=PyPlot.cm.jet )
  plot(t,log2.(coi2),"k")           # cone-of-influence, anything "below" is dubious
  contour(t, log2.(period2),sig95_2, [-99,1], colors="black")   # 95% significance value
  ylabel("period length")
  xlabel("time")
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)

end


function test_wcs_harelynx()
   # test wavelet cross spectrum  (WCS)

  # read in the data
  df = CSV.read("../data/Hare_Lynx.csv") 
  t   = df[:,1]  # time (year)
  h   = df[:,2]  # hare
  l   = df[:,3]  # lynx
  #
  # do the wavelet transform
  dt = 1    # sampling rate = 1 year
  s0 = 2.0  # starting scale = 1 year
  noctave = 4  # number of octaves
  nvoices = 100 # number of voices per octave

  # bi-wavelet analysis: cross-spectrum amplitude, phase, and coherence (wco)
  spower1, spower2, amplitude, phase, wco, period, scale, coi = 
    wcs(h,l,dt,noctave = noctave, s0=s0, nvoices = nvoices) 
    
  # calculate global CWS
  global_ws1 = mean(spower1,dims=2)  # global power spectrum hare
  global_ws2 = mean(spower2,dims=2)  # global power spectrum lynx
  global_wco = mean(wco,dims=2)      # global wco
  global_amplitude = mean(amplitude,dims=2)  # global wcs

  gws1 = global_ws1 / maximum(global_ws1)
  gws2 = global_ws2 / maximum(global_ws2)
  gwco = global_wco / maximum(global_wco)
  gwcs = global_amplitude / maximum(global_amplitude)
 
  # prefixed scale band
  # consider all scales within 0.6 octaves from peaks in global spectrum
  delta_ind = trunc(Int, 0.6*nvoices)
  ind_max   = argmax(gwcs)[1]        # peak scale in global WCS
  ind1 = ind_max - delta_ind  # indices  ...
  ind2 = ind_max + delta_ind
  period_min = period[ind1]  # .. and corresponding scales
  period_max = period[ind2]

  # identifiy phases at scales of maximal corr spectrum
  wcs_max_ind, angle_max, wco_max = get_maxInd(wco,phase,length(t),ind1,ind2)

  # line of maximal wcs
  wco_scale = log2.(period[wcs_max_ind])

  # circular statistics of phase distribution
  phi, var, deviation = circstats(pi*angle_max)
  println("Phase difference between hare and lynx (in pi) ", phi, 
          " std ", deviation)

  # calculate histogram of phase differences
  ph_bins  = 0.01 # phase bins = pi/100
  ph_width = 15    # histogram smoothed over 15 bins (=pi/8 = 27 deg)
  h_angle, h_bins = phasehist(angle_max, 1:length(angle_max),ph_bins,ph_width)


  # plotting scale range
  lminp = log2(minimum(period))
  lmaxp = log2(maximum(period))
  lrange = (trunc(lminp) :  trunc(lmaxp))
  # plotting time range
  tmax = maximum(t)
  tmin = minimum(t)


  # ********* Plotting ************
  fig_name = "Hare-lynx cross wavelet analysis" 
  nrows = 4
  if nrows == 4 # four columns
      fig = figure("fig_name",figsize=(10,8))   # 4 columns
      clf()
      fig_width_y       = 0.15
      figleft_x         = 0.1
      figleft_width_x   = 0.65
      figright_x        = 0.8
      figrigtht_width_x = 0.15
      fig_y1=0.79
      fig_y2=0.56
      fig_y3=0.33
      fig_y4=0.1
  end

  # Plot time series 
  ax = PyPlot.axes([figleft_x, fig_y1, figleft_width_x, fig_width_y])
  label="a"
  plot(t,h/maximum(h),"g",linewidth=2)
  plot(t,l/maximum(l),"r",linewidth=2)
  ylabel("abundance",labelpad=5, fontsize=12)
  xlim(tmin, tmax)
  ylim(0,1)
  yticks([0,0.5,1]) 
  ax1 = ax.transAxes
  text(-0.04, 1.10, label, fontsize=16, fontweight="bold", transform = ax1)
   
  # Plot phase plane
  ax = PyPlot.axes([figright_x, fig_y1, figrigtht_width_x, fig_width_y])
  label="b"
  time1 = 1900 - 1845 + 1 
  time2 = 1932 - 1845 + 1 
  hh = h[time1:time2]  # just for the phase plane ..
  ll = l[time1:time2]  # .. restrict data to nicely oscillating time window
  hh = hh / maximum(hh)
  ll = ll / maximum(ll)
  plot(hh, ll,"k")
  yticks([0,1])
  xticks([0,1])
  xlim(0,1)
  ylim(0,1)
  xlabel("hare", labelpad = -5, fontsize=12)  # labelpad: place the label a bit closer to axis
  ylabel("lynx", labelpad = -2, fontsize=12)
  ax1=ax.transAxes
  text(-0.14,1.10,label,fontsize=16, fontweight="bold",transform = ax1)
 
  # Plot wcs
  ax = PyPlot.axes([figleft_x, fig_y2, figleft_width_x, fig_width_y])
  label = "c"
  levels1 = range(0, stop=maximum(amplitude), length=200)
  contourf(t, log2.(period),amplitude,levels1, cmap=PyPlot.cm.jet )
  plot(t,log2.(coi),"k")           # cone-of-influence, anything "below" is dubious
  plot(t,wco_scale,"k-",linewidth=2)
  plot(t,ones(length(t))*log2(period_min),"k")    # lower border line
  plot(t,ones(length(t))*log2(period_max),"k")    # upper border line
  ylabel("period length")
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)
  xlim(tmin,tmax)
  ylabel("period length (y)",labelpad=9, fontsize=12)
  ax1 = ax.transAxes
  text(-0.04, 1.10, label, fontsize=16, fontweight="bold", transform = ax1)


  # Plot global wavelet cross spectrum 
  ax = PyPlot.axes([figright_x,fig_y2, figrigtht_width_x, fig_width_y])
  label = "d"
  plot(gwcs,log2.(period),"k")
  plot([0.,1], [1,1]*log2.(period_min),"k")
  plot([0.,1], [1,1]*log2.(period_max),"k")
  xlabel("power", fontsize=12)
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)
  xlim(0,1)
  xticks([0, 0.25, 0.5, 0.75, 1], [0., 0.25, 0.5, 0.75, 1.])
  ax1=ax.transAxes
  text(-0.14,1.10,label,fontsize=16, fontweight="bold",transform = ax1)


  # Plot wco
  ax = PyPlot.axes([figleft_x, fig_y3, figleft_width_x, fig_width_y])
  label = "e"
  levels1 = range(0, stop=maximum(wco), length=200)
  contourf(t, log2.(period),wco,levels1, cmap=PyPlot.cm.jet )
  plot(t,log2.(coi),"k")           # cone-of-influence, anything "below" is dubious
  plot(t,wco_scale,"k-",linewidth=2)
  plot(t,ones(length(t))*log2(period_min),"k")    # lower border line
  plot(t,ones(length(t))*log2(period_max),"k")    # upper border line
  ylabel("period length")
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)
  xlim(tmin,tmax)
  ylabel("period length (y)",labelpad=9, fontsize=12)
  ax1 = ax.transAxes
  text(-0.04, 1.10, label, fontsize=16, fontweight="bold", transform = ax1)


  # Plot global wavelet power spectra 
  ax = PyPlot.axes([figright_x,fig_y3, figrigtht_width_x, fig_width_y])
  label = "f"
  plot(gws1,log2.(period),"g")
  plot(gws2,log2.(period),"r")
  plot([0.,1], [1,1]*log2.(period_min),"k")
  plot([0.,1], [1,1]*log2.(period_max),"k")
  xlabel("power", fontsize=12)
  ylim(lmaxp,lminp)
  yticks(lrange, 2 .^ lrange)
  xlim(0,1)
  xticks([0, 0.25, 0.5, 0.75, 1], [0., 0.25, 0.5, 0.75, 1.])
  ax1=ax.transAxes
  text(-0.14,1.10,label,fontsize=16, fontweight="bold",transform = ax1)

  # Plot Phase difference
  ax = PyPlot.axes([figleft_x, fig_y4, figleft_width_x, fig_width_y])
  label = "g"
  plot(t,angle_max)
  plot([tmin, tmax],[0.0,0.0],color="lightgrey", "--") #, color="gray")
  plot([tmin, tmax],[0.5,0.5],color="lightgrey", "--") #, color="gray")
  plot([tmin, tmax],[1.0,1.0],color="lightgrey", "--") #, color="gray")
  xlim(tmin,tmax)
  yticks([-0.5, 0.5, 1.0])
  ylim(-0.5,1.0)
  xlabel("time in years", fontsize=12)
  ylabel(L"phase difference in $\pi$",labelpad=4, fontsize=12)
  ax1 = ax.transAxes
  text(-0.04, 1.12, label, fontsize=16,  fontweight="bold",transform = ax1)

  ax = PyPlot.axes([figright_x,fig_y4, figrigtht_width_x, fig_width_y], polar="true") # Create a polar axis
  label = "f"
  plot(h_bins, h_angle,"r")
  fill_between(h_bins,0, h_angle,color="red",alpha=0.5)
  xticks([0, pi/2, pi, 3*pi/2], ["0", L"$0.5\pi$", L"$\pi$", L"$1.5 \pi$" ])
  ax.tick_params(pad=-5)  # place the tick_labels a bit closer
  yticks([0,25,0.5,0.75],[])
  plot([0,0], [0,1], "g" , linewidth=2)   # reference line for algae
  ax1=ax.transAxes
  text(-0.3,1.10,label,fontsize=16, fontweight="bold",transform = ax1)


  savefig("hare_lynx.png")

end

test_wcs_harelynx()



end # module

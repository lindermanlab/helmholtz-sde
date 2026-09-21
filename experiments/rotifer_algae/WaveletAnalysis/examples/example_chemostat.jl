##  Test continuous wavelet transform 
##  with time series from predator-prey chemostat experiments
##
##  This file is a documented walk-through to reproduce
##  Extended Data Fig. 1 of the publication (associated 
##  to the data set)
##  Blasius et al. (2019) Long-term cyclic persistence in 
##  an experimental predator-prey system. 
## 
##  Author Bernd Blasius (bernd.blasius@gmail.com)

module W

using CSV, Dierckx, DSP, PyPlot, StatsBase

include("../src/cwt.jl")
include("../src/tools.jl")
include("../src/wcs.jl")

function do_wavelet(t,x)
  # calculate wavelet transform and power for the chemostat data
  # x: time series; dt: sampling rate
  dt  = 1           # sampling rate: 1 per day
  pad = 1           # pad the time series with zeroes (recommended)
  nvoice = 100      # this will do 100 sub-octaves per octave
  dj = 1 ./ nvoice
  s0 = 2.0          # this says start at a scale of 2 days
  noctave = 3       # calculate scales of 3 octaves
  j1      = noctave * nvoice
  mother  = "MORLET"

  x = x .- mean(x)    # normalize
  x = x ./ std(x)

  wave,period,scale,coi = wavelet(x,dt,pad,dj,s0,j1,mother)
  power = abs2.(wave)   # wavelet power
  # Global wavelet spectrum & significance levels:
  global_ws = mean(sqrt.(power),dims=2)   # time-average over all times

  wave, period, power, global_ws, scale, coi, nvoice
end



function ExtDataFig1()
    # load-in the data
    df = CSV.read("../data/chemostat_experiments/C1.csv")  # read-in data from C1
    tt  = df[:,1]  # time (days)
    mm  = df[:,2]  # algae (10^6 cells / ml)
    bb  = df[:,3]  # rotifers (animals / ml)
    err = df[:,4]  # egg ratio
    ee  = df[:,5]  # eggs      (eggs / ml)
    dd  = df[:,6]  # dead animals (dead animals / ml)
    ss  = df[:,7]   # nitrogen concentration in external medium (mumol N/l)

    # interpolate to constant sampling rate
    dt = 1   # sampling rate  (1 day)
    t,m  = sample_regular(mm,tt,dt,true)
    t,b  = sample_regular(bb,tt,dt,true)
    t,er = sample_regular(err,tt,dt,true)
    t,e  = sample_regular(ee,tt,dt,true)
    t,d  = sample_regular(dd,tt,dt,true)
     
    n = length(t)
    tmax = maximum(t)


    # continuous wavelet transform and wavelet power
    # wavalet parameters are already fitted for these chemoostat data
    (wave1, period1, power1, global_ws1, scale, coi, nvoice) = do_wavelet(t,m)
    (wave2, period2, power2, global_ws2, scale, coi, nvoice) = do_wavelet(t,b)
    (wave3, period3, power3, global_ws3, scale, coi, nvoice) = do_wavelet(t,er)
    (wave4, period4, power4, global_ws4, scale, coi, nvoice) = do_wavelet(t,e)
    (wave5, period5, power5, global_ws5, scale, coi, nvoice) = do_wavelet(t,d)

    # calculate wavelet cross spectrum (CWS)
    wcs2 = wave1 .* conj.(wave2)
    wcs3 = wave1 .* conj.(wave3)
    wcs4 = wave1 .* conj.(wave4)
    wcs5 = wave1 .* conj.(wave5)
    
    # time scale smoothing
    spower1 = smooth(power1,dt,nvoice,period2)
    spower2 = smooth(power2,dt,nvoice,period2)
    spower3 = smooth(power3,dt,nvoice,period2)
    spower4 = smooth(power4,dt,nvoice,period2)
    spower5 = smooth(power5,dt,nvoice,period2)
    swcs2   = smooth(wcs2,dt,nvoice,period2)
    swcs3   = smooth(wcs3,dt,nvoice,period2)
    swcs4   = smooth(wcs4,dt,nvoice,period2)
    swcs5   = smooth(wcs5,dt,nvoice,period2)

    swcs_abs2 = abs.(swcs2)   # absolute value of wavelet cross spectrum
    swcs_abs3 = abs.(swcs3)   # absolute value of wavelet cross spectrum
    swcs_abs4 = abs.(swcs4)   # absolute value of wavelet cross spectrum
    swcs_abs5 = abs.(swcs5)   # absolute value of wavelet cross spectrum
    sphs2 = angle.(swcs2)     # wavelet phase
    sphs3 = angle.(swcs3)     # wavelet phase
    sphs4 = angle.(swcs4)     # wavelet phase
    sphs5 = angle.(swcs5)     # wavelet phase

    # calculate wavelet coherence
    swco2 = swcs_abs2 ./ (sqrt.(spower1) .* sqrt.(spower2))
    swco3 = swcs_abs3 ./ (sqrt.(spower1) .* sqrt.(spower3))
    swco4 = swcs_abs4 ./ (sqrt.(spower1) .* sqrt.(spower4))
    swco5 = swcs_abs5 ./ (sqrt.(spower1) .* sqrt.(spower5))

    # calculate global CWS  ...
    global_wcso = mean(swco2,dims=2)      # time-average over all times
    global_wcs = mean(swcs_abs2,dims=2)   # time-average over all times
    # ... and normalize
    gws1 = global_ws1 / maximum(global_ws1)
    gws2 = global_ws2 / maximum(global_ws2)
    gwcso = global_wcso / maximum(global_wcso)
    gwcs = global_wcs / maximum(global_wcs)

    # prefixed scale band
    # consider all scales within 0.6 octaves from peaks in global spectrum
    delta_ind = trunc(Int, 0.6*nvoice)
    ind_max   = argmax(gwcs)[1]        # peak scale in global WCS
    ind_global1 = ind_max - delta_ind  # indices  ...
    ind_global2 = ind_max + delta_ind
    period_min = period2[ind_global1]  # .. and corresponding scales
    period_max = period2[ind_global2]

    # find strongest co-oscillating component
    # i.e., maximal cross-spectrum at each time instance
    wcs_max_ind2,angle_max2,wco_max2 = get_maxInd(swco2,sphs2,n,ind_global1,ind_global2)
    wcs_max_ind3,angle_max3,wco_max3 = get_maxInd(swco3,sphs3,n,ind_global1,ind_global2)
    wcs_max_ind4,angle_max4,wco_max4 = get_maxInd(swco4,sphs4,n,ind_global1,ind_global2)
    wcs_max_ind5,angle_max5,wco_max5 = get_maxInd(swco5,sphs5,n,ind_global1,ind_global2)

    # Identifiy coherent oscillation regions
    cv_wco = 0.83    # --> 95% significance level according to Monte-Carlo simulations
    indices_wco = findall(x-> x >= cv_wco, wco_max2)
    
    # line of maximal wcs
    wco_scale = log2.(period2[wcs_max_ind2])

    # remove points outside coi
    relevant = wco_scale .< log2.(coi)
    indices_wco = indices_wco[findall(x-> relevant[x], indices_wco)]

    # remove points where maximal wco is at border of scale range
    indices_wco = indices_wco[findall(x-> (x != ind_global2 && x != ind_global1)
                               , wcs_max_ind2[indices_wco])]

    # find contiguous time windows of correlated oscillations 
    iwco = indices_wco
    # find all breaks, ie difference of 2 indices larger than 1 ..
    inds = findall(x->x>1, diff(iwco))  
    # .. and then put the indices of oscillation times together
    strt_inds = vcat(iwco[1], iwco[inds .+ 1])
    stop_inds = vcat(iwco[inds .- 1] .+ 1, iwco[end])
    delta_inds = stop_inds .- strt_inds .+ 1

    # calculate mean angles and standard deviation
    phi2, var2, std2 = circstats(pi*angle_max2[indices_wco])
    phi3, var3, std3 = circstats(pi*angle_max3[indices_wco])
    phi4, var4, std4 = circstats(pi*angle_max4[indices_wco])
    phi5, var5, std5 = circstats(pi*angle_max5[indices_wco])
    fac = 180.0/pi
    println("mean angle differences (in degrees)")
    println("angle 2 ", fac*phi2, " std ", fac*std2)
    println("angle 3 ", fac*phi3, " std ", fac*std3)
    println("angle 4 ", fac*phi4, " std ", fac*std4)
    println("angle 5 ", fac*phi5, " std ", fac*std5)

    # calculate histograms of phase differences
    ph_bins  = 0.01  # phase bins = pi/100
    ph_width = 15    # histogram smoothed over 15 bins (=pi/8 = 27 deg)
    h_angle2, h_bins2 = phasehist(angle_max2, indices_wco, ph_bins, ph_width)
    h_angle3, h_bins3 = phasehist(angle_max3, indices_wco, ph_bins, ph_width)
    h_angle4, h_bins4 = phasehist(angle_max4, indices_wco, ph_bins, ph_width)
    h_angle5, h_bins5 = phasehist(angle_max5, indices_wco, ph_bins, ph_width)

    # standardize phase histograms for plotting
    max_hist_value = maximum([maximum(h_angle2), maximum(h_angle3), maximum(h_angle4),
    maximum(h_angle5)])
    h_angle2 /= max_hist_value
    h_angle3 /= max_hist_value
    h_angle4 /= max_hist_value
    h_angle5 /= max_hist_value

    # calculate histogram of relevant period length
    period_hist = fit(Histogram, wco_scale[indices_wco], nbins=12)
    per_hists = period_hist.weights
    per_hists_bins = period_hist.edges[1][2:end]
    
    # repeat the last histogram value at the end to get a smooth polar diagram
    h_bins2 = collect(h_bins2)
    push!(h_bins2, h_bins2[1]+ 2*pi)
    push!(h_angle2, h_angle2[1])
    h_bins3 = collect(h_bins3)
    push!(h_bins3, h_bins3[1]+ 2*pi)
    push!(h_angle3, h_angle3[1])
    h_bins4 = collect(h_bins4)
    push!(h_bins4, h_bins4[1]+ 2*pi)
    push!(h_angle4, h_angle4[1])
    h_bins5 = collect(h_bins5)
    push!(h_bins5, h_bins5[1]+ 2*pi)
    push!(h_angle5, h_angle5[1])

    # plotting scale range
    lminp = log2(minimum(period1))
    lmaxp = log2(maximum(period1))
    lrange = (trunc(Int, lminp) :  trunc(Int, lmaxp))

    
    # ********* Plotting ************
    figure("ExtDataFig1",figsize=(10,10))  # 6 columns
    clf()
    figleft_width_x = 0.65
    figrigth_width_x = 0.15
    figleft_x = 0.1
    figright_x = 0.8
    fig_width_y = 0.1
    fig_sep_y   = 0.06
    fig_y6=0.06
    fig_y5=fig_y6 + fig_width_y + fig_sep_y
    fig_y4=fig_y5 + fig_width_y + fig_sep_y
    fig_y3=fig_y4 + fig_width_y + fig_sep_y
    fig_y2=fig_y3 + fig_width_y + fig_sep_y
    fig_y1=fig_y2 + fig_width_y + fig_sep_y


    # ---------------------------------
    # Time series
    ax = PyPlot.axes([figleft_x, fig_y1, figleft_width_x, fig_width_y])
    plot(t,m/maximum(m),"g",linewidth=2)
    ylabel("abundance", labelpad=8, fontsize=10)
    plot(t,b/maximum(b),"r",linewidth=2)
    xlim(0, tmax)
    ylim(0,1)
    yticks([0,0.5,1])
    ax1 = ax.transAxes
    text(-0.04, 1.10, "a", fontsize=16, fontweight="bold", transform = ax1)

    # --------------------------------------------
    # phase plane
    ax = PyPlot.axes([figright_x, fig_y1, figrigth_width_x, fig_width_y])
    time1=90; time2=130
    # do a bit of filtering for the phase-portrait plot
    mfit = preprocess1(t,m,dt)
    bfit = preprocess1(t,b,dt)
    mf = mfit[time1:time2]   #restrict data to time window
    bf = bfit[time1:time2]
    mf = mf .+ abs(minimum(mf))
    bf = bf .+ abs(minimum(bf))
    mf = mf / maximum(mf)
    bf = bf / maximum(bf)
    plot(mf, bf)
    arrow(0.57,0.9,-0.1,-0.02,color="grey", width=0.02)   # thin arrow beside the line
    yticks([0,1])
    xticks([0,1])
    xlim(0,1)
    ylim(0,1)
    xlabel("prey", labelpad = -5, fontsize=10)  # labelpad: place the label a bit closer to axis
    ylabel("predator", labelpad = -2, fontsize=10)
    ax1=ax.transAxes
    text(-0.14,1.10,"b",fontsize=16, fontweight="bold",transform = ax1)


    # --------------------------------------------
    # power spectrum algae
    ax = PyPlot.axes([figleft_x, fig_y2, figleft_width_x, fig_width_y])
    pcolormesh(t, log2.(period1),log2.(spower1), cmap=PyPlot.cm.jet)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,tmax)
    plot(t,ones(length(t))*log2(period_min),"k")    # lower border line
    plot(t,ones(length(t))*log2(period_max),"k")    # upper border line
    plot(t,log2.(coi),"k")
    plot(t,log2.(period2[wcs_max_ind2]),"k")    # line of maximal wcs
    ylabel("period length (d)", labelpad=12, fontsize=10)
    ax1 = ax.transAxes
    text(-0.04, 1.10, "c", fontsize=16, fontweight="bold", transform = ax1)

    # --------------------------------------------
    # global power spectrum
    ax = PyPlot.axes([figright_x, fig_y2, figrigth_width_x, fig_width_y])
    plot(gws1,log2.(period2),"g")
    plot([0.,1], [1,1]*log2.(period_min),"k")
    plot([0.,1], [1,1]*log2.(period_max),"k")
    xlabel("power", fontsize=10)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,1)
    xticks([0, 0.25, 0.5, 0.75, 1])
    ax1=ax.transAxes
    text(-0.14,1.10,"d",fontsize=16, fontweight="bold",transform = ax1)

    # --------------------------------------------
    # power spectrum rotifers
    ax = PyPlot.axes([figleft_x, fig_y3, figleft_width_x, fig_width_y])
    pcolormesh(t, log2.(period1),log2.(spower2), cmap=PyPlot.cm.jet)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,tmax)
    plot(t,ones(length(t))*log2(period_min),"k")    # lower border line
    plot(t,ones(length(t))*log2(period_max),"k")    # upper border line
    plot(t,log2.(coi),"k") #cone-of-influence, anything "below" is dubious
    plot(t,log2.(period2[wcs_max_ind2]),"k")    # line of maximal wcs
    ylabel("period length (d)", labelpad=12, fontsize=10)
    ax1 = ax.transAxes
    text(-0.04, 1.10, "e", fontsize=16, fontweight="bold", transform = ax1)

    # --------------------------------------------
    # global power spectrum
    ax = PyPlot.axes([figright_x, fig_y3, figrigth_width_x, fig_width_y])
    plot(gws2,log2.(period2),"r")
    plot([0.,1], [1,1]*log2.(period_min),"k")
    plot([0.,1], [1,1]*log2.(period_max),"k")
    xlabel("power", fontsize=10)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,1)
    xticks([0, 0.25, 0.5, 0.75, 1])
    ax1=ax.transAxes
    text(-0.14,1.10,"f",fontsize=16, fontweight="bold",transform = ax1)

    # --------------------------------------------
    # wavelet cross spectrum
    ax = PyPlot.axes([figleft_x, fig_y4, figleft_width_x, fig_width_y])
    label = "g"
    pcolormesh(t, log2.(period1),log2.(swcs_abs2), cmap=PyPlot.cm.jet)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,tmax)
    plot(t,ones(length(t))*log2(period_min),"k")    # lower border line
    plot(t,ones(length(t))*log2(period_max),"k")    # upper border line
    plot(t,log2.(coi),"k") #cone-of-influence, anything "below" is dubious
    plot(t,log2.(period2[wcs_max_ind2]),"k")    # line of maximal wcs
    ylabel("period length (d)", labelpad=12, fontsize=10)
    ax1 = ax.transAxes
    text(-0.04, 1.10, label, fontsize=16, fontweight="bold", transform = ax1)

    # --------------------------------------------
    # global power spectrum
    ax = PyPlot.axes([figright_x,fig_y4, figrigth_width_x, fig_width_y])
    label = "h"
    plot(gwcs,log2.(period2),"k")
    plot([0.,1], [1,1]*log2.(period_min),"k")
    plot([0.,1], [1,1]*log2.(period_max),"k")
    xlabel("power", fontsize=10)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,1)
    xticks([0, 0.25, 0.5, 0.75, 1])
    ax1=ax.transAxes
    text(-0.14,1.10,label,fontsize=16, fontweight="bold",transform = ax1)


    # --------------------------------------------
    # plot WCO
    ax = PyPlot.axes([figleft_x, fig_y5, figleft_width_x, fig_width_y])
    label = "i"
    #pcolormesh(t, log2.(period1),swco2, cmap=PyPlot.cm.jet, alpha = 0.5)
    levels = range(0, stop=1, length=200)
    contourf(t, log2.(period1),swco2,levels, cmap=PyPlot.cm.jet )
    contour(t, log2.(period1),swco2,[cv_wco], colors="black")
    plot(t,log2.(coi),"k") #cone-of-influence, anything "below" is dubious
    plot(t,ones(length(t))*log2(period_min),"k")    # lower border line
    plot(t,ones(length(t))*log2(period_max),"k")    # upper border line
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,tmax)
    for i=1:length(strt_inds)
      i1 = strt_inds[i]; i2 = stop_inds[i]
      plot(t[i1:i2],wco_scale[i1:i2],"k-",linewidth=3)
    end
    ylabel("period length (d)", labelpad = 12, fontsize=10)
    ax1 = ax.transAxes
    text(-0.04, 1.10, label, fontsize=16, fontweight="bold", transform = ax1)


    # --------------------------------------------
    # plot histogram of relevant period length
    ax = PyPlot.axes([figright_x,fig_y5, figrigth_width_x, fig_width_y])
    label = "j"
    per_hists = per_hists / maximum(per_hists)  
    mmm = maximum(per_hists)
    plot([0.,mmm], [1,1]*log2.(period_min),"k")
    plot([0.,mmm], [1,1]*log2.(period_max),"k")
    wd = 0.9*per_hists_bins[3]-per_hists_bins[2]
    barh(per_hists_bins,per_hists,height=wd,align="center",alpha=0.6)
    xlabel("histogram", fontsize=10)
    ylim(lmaxp,lminp)
    yticks(lrange, 2 .^ lrange)
    xlim(0,1)
    xticks([0, 0.25, 0.5, 0.75, 1])
    ax1=ax.transAxes
    text(-0.14,1.10,label,fontsize=16, fontweight="bold",transform = ax1)


    # --------------------------------------------
    ax = PyPlot.axes([figleft_x, fig_y6, figleft_width_x, fig_width_y])
    label = "k"
    plot([0, tmax],[0.0,0.0],color="grey", "--") #, color="gray")
    plot([0, tmax],[0.5,0.5],color="grey", "--") #, color="gray")
    plot([0, tmax],[1.0,1.0],color="grey", "--") #, color="gray")
    plot(t,angle_max2,"--", color="grey")
    plot(t,angle_max3,"--", color="grey")
    plot(t,angle_max4,"--", color="grey")
    plot(t,angle_max5,"--", color="grey")
    for i=1:length(strt_inds)
      i1 = strt_inds[i]; i2 = stop_inds[i]
      plot(t[i1:i2],angle_max2[i1:i2],"r-",linewidth=2)
      plot(t[i1:i2],angle_max3[i1:i2],"b-",linewidth=2)
      plot(t[i1:i2],angle_max4[i1:i2],"k-",linewidth=2)
      plot(t[i1:i2],angle_max5[i1:i2],"y-",linewidth=2)
    end
    xlim(0,tmax)
    yticks([-0.5, 0, 0.5, 1.0, 1.5])
    ylim(-0.5,1.5)
    xlabel("time in d", fontsize=10)
    ylabel(L"phase difference in $\pi$", labelpad = -2, fontsize=10)
    ax1 = ax.transAxes
    text(-0.04, 1.10, label, fontsize=16,  fontweight="bold",transform = ax1)


    # --------------------------------------------
    ax = PyPlot.axes([figright_x,fig_y6, figrigth_width_x, fig_width_y], polar="true") # Create a polar axis
    label = "l"
    plot(h_bins2, h_angle2,"r")
    fill_between(h_bins2,0, h_angle2,color="red",alpha=0.5)
    plot(h_bins3, h_angle3,"b")
    fill_between(h_bins3,0, h_angle3,color="blue",alpha=0.5)
    plot(h_bins4, h_angle4,"k")
    fill_between(h_bins4,0, h_angle4,color="black",alpha=0.5)
    plot(h_bins5, h_angle5,"y")
    fill_between(h_bins5,0, h_angle5,color="yellow",alpha=0.5)
    xticks([0, pi/2, pi, 3*pi/2], ["0", L"$0.5\pi$", L"$\pi$", L"$1.5 \pi$" ])
    ax.tick_params(pad=-5)  # place the tick_labels a bit closer
    yticks([0,25,0.5,0.75],[])
    plot([0,0], [0,1], "g" , linewidth=2)   # reference line for algae
    ax1=ax.transAxes
    text(-0.57,1.10,label,fontsize=16, fontweight="bold",transform = ax1)

    #savefig("ExtDataFig1.png")
    println("done")

end

ExtDataFig1()


end # modul

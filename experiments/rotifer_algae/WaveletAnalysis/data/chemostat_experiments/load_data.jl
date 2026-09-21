## Julia code to load-in chemostat data

using CSV, DataFrames, Dierckx, PyPlot

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


function get_data(exp_code, make_regular)
    # load-in data from csv-file
    # exp_code : string coding the experimental trial ("C1" to "C10")
    # make_regular : flag
    #                  true  --> interpolate to regular time spacing (needed for wavelets)
    #                  false --> keep the raw data

    forced = false
    if exp_code == "C8" || exp_code == "C9"  # these are the externally forced systems
        forced = true
    end

    # load-in the data
    df = CSV.read(exp_code * ".csv")  # read in data
    t  = df[:,1]  # time (days)
    m  = df[:,2]  # algae (10^6 cells / ml)
    b  = df[:,3]  # rotifers (animals / ml)
    er = df[:,4]  # egg ratio
    e  = df[:,5]  # eggs      (eggs / ml)
    d  = df[:,6]  # dead animals (dead animals / ml)
    s = df[:,7]   # nitrogen concentration in external medium (mumol N/l)


    if make_regular == false   # either return the raw data  ....
        return forced,t,m,b,er,e,d,s
    end


    # ... else, do some minimal pre-processing

    # eliminate missing values, this yields new time vectors for every state
    M,tM = deal_missing(m,t)
    B,tB = deal_missing(b,t)
    E,tE  = deal_missing(e,t)
    D,tD  = deal_missing(d,t)
    ER,tER  = deal_missing(er,t)
    S,tS  = deal_missing(s,t)

    # interpolation to a regular one-day sampling interval
    t_end = trunc(t[end])
    tt = 0:t_end
    mm = evaluate(Spline1D(tM,M),tt)
    bb = evaluate(Spline1D(tB,B),tt)
    ee = evaluate(Spline1D(tE,E),tt)
    er = evaluate(Spline1D(tER,ER),tt)
    dd = evaluate(Spline1D(tD,D),tt)

    # the splines artificially create some smoothing error 
    # so we need to correct this
    # this is a simple hack, but works..
    if forced
        ss = evaluate(Spline1D(tS,S),tt)
        for i=1:length(ss)  # spline artifacts are largest for the external forcing signal
            if ss[i] < 100
                ss[i] = 0.0  
            else
                ss[i] = 160.0
            end
        end
    else
        ss = S[1] * ones(length(tt))  # these are constants, so no splines needed
    end
    # if interpolation artificially created small negative values, we remove them
    mm = minimum(mm) < 0 ? mm .- minimum(mm) : mm
    bb = minimum(bb) < 0 ? bb .- minimum(bb) : bb
    ee = minimum(ee) < 0 ? ee .- minimum(ee) : ee
    er = minimum(er) < 0 ? er .- minimum(er) : er
    dd = minimum(dd) < 0 ? dd .- minimum(dd) : dd
    if forced
        ss = minimum(ss) < 0 ? ss .- minimum(ss) : ss
    end

    
    return forced,tt,mm,bb,er,ee,dd,ss   
end


function test()
    # simple test code to load data and make a plot
    exp_code = "C10"
    make_regular = true
    forced, t, m, b, er, e, d, s = get_data(exp_code, make_regular)

    figure(exp_code)
    clf()
    subplot(3,2,1)
    plot(t,m)

    ylabel("algae")
    subplot(3,2,2)
    plot(t,b)
    ylabel("rotifers")
    subplot(3,2,3)
    plot(t,er)
    ylabel("egg-ratio")
    subplot(3,2,4)
    plot(t,e)
    ylabel("eggs")
    subplot(3,2,5)
    plot(t,d)
    ylabel("dead")
    xlabel("time")
    subplot(3,2,6)
    plot(t,s)
    ylabel("external nutrients")
    xlabel("time")

end


test()


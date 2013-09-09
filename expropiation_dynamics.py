# Numerical Analysis for:
#
# Aguiar, Mark, Manuel Amador, and Gita Gopinath. 2009. 
# "Expropriation Dynamics." American Economic Review P&P, 99(2): 473-79.
#
# Python 2.5 code written by Manuel Amador, 2008
# (updated to Python 2.7)
#
# Requires libraries: Scipy, Numpy and Matplotlib 
#
#
# ----------------- Basic Python Configuration --------------------
from __future__ import division  # correct numerical division 
from matplotlib import pyplot as plt
from numpy import *              # numerical (matrices) tools 
from scipy import optimize       # numerical (optimization) tools 
from time import time            # timer tools

# ----------------- Some Generic Functions ------------------------
def interp1(x, XY):
    """My simple linear interpolating function.
    Argument x must be a one dimensional array or
    a scalar.  Argument XY is a matrix of size
    two by n, with the first row is in increasing order.
    The function extrapolates.""" 
    pos = XY[0].searchsorted(x) - 1
    pos =  pos*(pos > 0) - 1 * (pos == len(XY[0]) - 1)
    if len(XY[0]) != 1:
        ans = XY[1][pos] + ((XY[1][pos + 1]-XY[1][pos])/(XY[0][pos + 1] -
          XY[0][pos])) * (x - XY[0][pos])
    return ans if len(XY[0]) !=1 else array([XY[1][0]])

def get_slopes(x):
    "Returns the slopes of a two by n matrix"
    return diff(x)[1] / diff(x)[0]

def convexify(x, convex=True):
    """Returns the convex fronteir of a function x[0] -> x[1].
    If convex is false, then returns the concave fronteir"""
    cond = (lambda x : diff(x) <= 0) if convex else lambda x : diff(x) >= 0
    while True:
        slopes = get_slopes(x)
        auxmask = where(cond(slopes))
        if auxmask[0].any():
            x = delete(x, auxmask[0] + 1, axis = 1)
        else: break
    return x

#---------- More functions : dependent of parameters ----------
def f(k, z):
    "Production function"
    return  (k ** a) * z  + a0 * z

def Ef(k):
    "Expected production function"
    return pH * f(k, zH) + pL * f(k, zL)

def finverse(x, z):
    "Inverse of production function"
    return ((x - a0 * z) / z) ** (1 / a)

def fprime (k, z):
    "Derivative of production function"
    return a * z * (k) ** (a - 1) 

def fprimeinverse (x, z):
    "Derivative of production function"
    return  (x / (z * a)) ** (1 / (a - 1))

def U(c):
    "Utility function"
    return (c ** (1 - rho)) / (1 - rho) if rho != 1 else log(c)

def Uprime(c):
    "Derivative of utility function"
    return c ** (-rho) if rho != 1 else 1 / c

def C(x):
    "Inverse of U"
    return ((1 - rho) * x) ** (1 / (1 - rho)) if rho != 1 else exp(x)

def Cprime(x):
    "Derivative of inverse U"
    return ((1 - rho) * x) ** (1 / (1 - rho) - 1) if rho != 1 else exp(x)

def Cprimeinverse(x):
    "Inverse of the derivative of the inverse of U"
    return x ** ( 1 / (1 / (1 - rho) - 1)) / (1 - rho) \
           if rho != 1 else log(x)

def compute_first_best():
    "Computes the first best fronteir"
    return vstack((linspace(vmin, vmax, 10000) ,
                   ((R * (Ef(kstar) - (r + d) * kstar)/ 
                     r - (1 / (1 - be)) * ((bR) ** (-be / (1 - be))) * 
                     exp((1-be) * linspace(vmin, vmax, 10000)))
                    if rho == 1 else R * (Ef(kstar) - (r + d) * kstar) /
                    r - (1 - be * (bR) ** ((1 - rho) / rho)) **
                    (rho / (1 - rho)) * ((1 - rho) *
                                         linspace(vmin, vmax, 10000)) **
                    (1 / (1 - rho)))))
    
def do_main_iter(B, do_plots=True,
                 fast_iters=10, stop_iter=10**(-5), klength=10000,
                 max_iters=200):
    """Does the main iteration of the value function starting from arg B.
    If none provided, starts from first best frontier"""

    def foc(v,  z, boundary = vmax):
        """Local function that returns the value of w that solves the
        first order condition for z and a value for the deviation
        v = h(k) given that the participation constraint is
        binding.
        Note: it restricts the solution to be above vmin and
        below boundary."""
        return boundary if (Cprime(v -  be * boundary)  >= - 
                            interp1(boundary, Bslopes)
                            / bR) \
                        else (vmin if (Cprime(v -  be * vmin) <=
                                       - interp1(vmin , Bslopes)
                                        / bR)
                              else optimize.brentq(lambda w: Cprime(v - be * w)
                                                   + (1 / bR) *
                                                   interp1(w,
                                                                      Bslopes)
                                                   , vmin, boundary))

    BbeforeJudd = B.copy()
    t1 = time()                     # Initialize some timers.
    kgrid = linspace(kstar, 0, klength) # capital grid
    # Storing some computations    
    fgrid_H = f(kgrid, zH)              
    fgrid_L = f(kgrid, zL)
    Vautgrid_H = U(fgrid_H) + be * A0
    Vautgrid_L = U(fgrid_L) + be * A0
    lambdakgrid_H =  (pH * fprime(kgrid, zH) + pL *
                      fprime(kgrid, zL) - (r + d)) / (Uprime(fgrid_H)
                                                      * fprime(kgrid, zH))
    if lambdakgrid_H[1]-lambdakgrid_H[0] > 0:
        print "Watch out!! Lambda(k) is increasing at k = kaut"
        print "----->>> Possible convexity violation <<<------"
    if do_plots:
        plt.figure()
        plt.plot(kgrid, lambdakgrid_H, '.-')
        plt.title('plot of lambdak: should be monotone')
        plt.figure()
    allpcbind = False
    for main_iter in range(max_iters):
        # Outer loop: value function iteration.
        # This iterates until max_iters or convergence is achieved.
        # Should converge before max_iters!
        print "iter: ", main_iter, '-------------------------------------'
        if do_plots:
            plt.plot(B[0, :], B[1, :], '.-')
            plt.title('value function iteration')
        Bslopes = vstack((B[0, :-1], get_slopes(B))) # Get the slopes of B
        Binvertslopes = vstack((Bslopes[1,::-1], Bslopes[0,::-1]))
        v = []
        BB = []
        policies = array([0, 0, 0, 0, 0])
        wH = wL = wmax = vmax
        vmin = B[0, 0]
        allpcbind = False
        for k, fH, fL, VH, VL, lambdaH in zip(kgrid, fgrid_H, fgrid_L,
                                              Vautgrid_H, Vautgrid_L,
                                              lambdakgrid_H):
            # Inner loop: iterates downwards from kstar.
            wH = foc(VH, wH)
            uH = VH - be * wH
            if not allpcbind:
                muH = Cprime (uH) - lambdaH / pH
                if muH >= 0:
                    uL = Cprimeinverse(muH)
                    wL = max(interp1(- muH * bR,
                                                Binvertslopes), vmin)
                    if uL + be * wL < VL:
                        allpcbind = True
                else:
                    allpcbind = True
            if allpcbind:
                wL = foc(VL, wL)
                uL = VL - be * wL
            # Stores the calculations
            v.append(pH * (uH + be * wH) + pL * (uL + be * wL))
            BB.append(pH * fH + pL * fL - (r + d) * k + pH *
                      (- C(uH) + interp1(wH, B) / R ) +
                      pL * (- C(uL) + interp1(wL, B) / R))
            policies = vstack((array([k, uL, uH, wL, wH]), policies))
            if k < kstar:
                if BB[-1] < BB[-2]:
                    # If the value decreased, we stop iterating.
                    BB[-1] = BB[-2]
                    policies[-1] = policies[-2]
                    v[-1] = v[-2] - (v[-3] - v[-2])
                    break
        policies = policies[:-1,:]  # Eliminate the dummy row in policy
        v.reverse()                 # and reverse the list.
        BB.reverse()
        B = array([v, BB])          # Stores the value function. 
        if do_plots: plot(v, BB,'.-')
        # Compute the difference between the new value function and the old one
        #                                     (before Judd accelerator) ^^^^^^^
        distance = abs(interp1(B[0,:], BbeforeJudd) - B[1,:]).max()
        BbeforeJudd = B
        print 'Error bound:', round(distance*(1-be), 8), \
              'time:', round(time() - t1,2), 'seconds'
        t1 = time()  
        if  distance < stop_iter:
            print '-> iteration converged: ', distance*(1-be)
            break
        Cl = C(policies[:,1])      # Stores some computations.
        Ch = C(policies[:,2])      # ^^^ ditto 
        Efpol = pL * f(policies[:,0], zL) + pH * f(policies[:,0], zH)
        for counter in range(fast_iters):
            # Do Judd's iterations using policy functions.
            BB = (Efpol - (r + d) * policies[:,0] +
                  pL*(-Cl + interp1(policies[:,3],
                                               array([B[0], BB]))/R)+
                  pH*(-Ch + interp1(policies[:,4],
                                               array([B[0], BB]))/R))
            if do_plots:
                plt.plot(v, BB,'-')
        if min(BB) < BB[-1]:
            # makes sure the value is decreasing
            BB = ones(size(BB)) * BB[-1]
        B = convexify(vstack((B[0],BB)), convex = False)
        # No reason for Judd's accelerator to return
        # a concave fn, so force it to be concave.
    return B, policies

def do_plots(B, policies, k):
    "Generates some plots for a given simulation of the model."
    plt.figure()
    plt.subplot(221)
    plt.plot(B[0,:], B[1,:], '.')
    plt.grid(True)
    plt.title('value function B(v)')
    plt.subplot(222)
    plt.plot(B[0,:], policies[:,0],'.')
    plt.grid(True)
    plt.title('k versus v')
    plt.subplot(223)
    plt.plot(B[0,:], C(policies[:,1]),'.')
    plt.plot(B[0,:], C(policies[:,2]),'.')
    plt.grid(True)
    plt.title('c versus v')
    plt.subplot(224)
    plt.plot(B[0,:], policies[:,3],'.-')
    plt.plot(B[0,:], policies[:,4],'.-')
    plt.plot(B[0,:], B[0,:],'-')
    plt.grid(True)
    plt.title('w versus v')
    plt.figure()
    plt.plot(k[:100], '.-') # plots a path
    plt.plot(k[:100], kstar * ones(100), '-') 
    plt.title('a simulated path for k')

def compute_path(B, policies, path_length = 10000, shock_path=None):
    """Computes a randome path. It returns a path of shock indexes (0 or 1),
    the actual values of the shocks, the promised value path,
    and the capital path"""
    if shock_path == None:
        shock_path = random.binomial(1, pH, path_length)
    v = [B[0, -1]]
    Bpol = [array([B[0], policies[:, 3]]),
            array([B[0], policies[:, 1 + 3]])]    # Stores to speed things up
    for z in shock_path:
        v.append(interp1(v[-1], Bpol[z]))
    v = array(v)
    k = interp1(v, array([B[0], policies[:, 0]]))
    c = zeros(size(shock_path))
    c[shock_path == 1] = C(interp1(v[shock_path == 1],
                                              array([B[0], policies[:, 2]])))
    c[shock_path == 0] = C(interp1(v[shock_path == 0],
                                              array([B[0], policies[:, 1]])))
    return shock_path, shock_path * zH + (1 - shock_path) * zL, v, k, c

def simulate_stats(sims, path_length=10000):
    """Compute statistics from models inside the dictionary sims
    where the key is beta. Returns a dictionary with the statistics."""
    std_i = [];  std_lny = []; std_lnc = [];  ac_lny = []; std_deltalny = []
    belist = []; ratio_std_lnc_lny = []; corr_nx_lny = []; mean_k_y = []
    corr_nxy_lny = [];  corr_nxy_y = []; mean_b_y = []; mean_i_y = []
    ratio_std_lni_lny = []; mean_k_kstar = []; corr_deltaby_lny = []
    corr_deltab_lny = []; ac_nxy = []
    # shock_path = binomial(1, pH, path_length) 
    for (bb, simul), contador in zip(sims.items(), range(size(sims.keys()))):
        print contador, ", ", bb
        st, zt, vt, kt, ct = compute_path(simul['Bv'], simul['pol'],
                                          path_length)
        it = kt[1:] - (1 - d) * kt[:-1]
        kt = kt[:-1]
        vt = vt[:-1]        
        Yt = f(kt, zt)
        Bt = interp1(vt, simul['Bv'])
        nxt = (Yt - ct - it)        
        nxst = (Yt - ct - it)/Yt
        ct = log(ct)
        yt = log(Yt)
        mean_b_y.append(mean(Bt / (R * Yt)))
        mean_k_y.append(mean(kt / Yt))
        mean_i_y.append(mean(it / Yt))
        mean_k_kstar.append(mean(kt / kstar))
        std_i.append(std(it))
        std_lny.append(std(yt))
        std_deltalny.append(std(yt[1:] - yt[:-1]))
        std_lnc.append(std(ct))
        ratio_std_lnc_lny.append(std(ct)/std(yt))
        ratio_std_lni_lny.append(std(log(it[100:]))/std(yt[100:]))
        ac_lny.append(corrcoef(yt[1:],yt[:-1])[0,1])
        corr_nxy_lny.append(corrcoef(nxst, yt)[0,1])
        corr_nx_lny.append(corrcoef(nxt, yt)[0,1])
        corr_nxy_y.append(corrcoef(nxst, Yt)[0,1])
        ac_nxy.append(corrcoef(nxst[1:], nxst[:-1])[0,1])
        corr_deltab_lny.append(corrcoef((Bt[1:]-Bt[:-1]), yt[:-1])[0,1])
        corr_deltaby_lny.append(corrcoef((Bt[1:]-Bt[:-1])/Yt[:-1], yt[:-1])[0,1])
        belist.append(bb)
    return {'belist': [belist, r'$\beta$'],
            'std_lny': [std_lny, r'Standard Deviation of $\log(y)$'],
            'std_lnc': [std_lnc, r'Standard Deviation of $\log(c)$'],
            'std_i': [std_i, r'Standard Deviation of $i$'],
            'ac_lny': [ac_lny, r'Autocorrelation Coefficient of $\log(y)$'],
            'std_deltalny': [std_deltalny, 
                             r'Standard Deviation of $\Delta(\log(y))$'],
            'ratio_std_lnc_std_lny': [ratio_std_lnc_lny,
                   r'Ratio of Standard Deviations of $\log(c)$ to $\log(y)$'],
            'ratio_std_lni_std_lny': [ratio_std_lni_lny,
                   r'Ratio of Standard Deviations of $\log(i)$ to $\log(y)$'],
            'corr_nxy_lny': [corr_nxy_lny,
                  r'Correlation Coefficient of $\frac{nx}{y}$ with $\log(y)$'],
            'corr_nxy_y': [corr_nxy_y,
                  r'Correlation Coefficient of $\frac{nx}{y}$ with $Y$'],
            'corr_nx_lny': [corr_nx_lny,
                  r'Correlation Coefficient of $nx$ and $\log(y)$'],
            'mean_i_y': [mean_i_y,
                  r'Mean of $\frac{i}{y}$'],
            'mean_k_y': [mean_k_y,
                  r'Mean of $\frac{k}{y}$'],
            'mean_b_y': [mean_b_y,
                  r'Mean of Debt to Output Ratio'],
            'mean_k_kstar': [mean_k_kstar,
                  r'Mean of $\frac{k}{k^\star}$'],
            'corr_deltab_lny': [corr_deltab_lny,
                  r'Correlation of $B_{t+1} - B_{t}$ with $Y_t$ '],
            'corr_deltaby_lny': [corr_deltaby_lny,
                  r'Correlation of $\frac{B_{t+1} - B_{t}}{Y_t}$ with $Y_t$ '],
            'ac_nxy': [ac_nxy, r'Autocorrelation of $nx/y$']}

def do_stats_plots(simuls, fname="plot", save_plots=False, titles=True,
                   mystatskeys=None, exten="eps"):
    """Plots the statistics, where simulstats is a dictionary with the
    statistics, of the form returned by simulate_stats()"""
    beRlist = array(simuls['belist'][0]) *  R
    itemstoiter = simuls.keys() if mystatskeys == None else mystatskeys
    for argu in itemstoiter:
        if argu != 'belist':
            plt.figure()
            dd = zip(beRlist, simuls[argu][0])
            dd.sort()
            plt.plot(array(dd)[:,0], array(dd)[:,1], 'o-')
            plt.xlabel(r'$\beta (1+r)$')
            if titles:
                plt.title(simuls[argu][1], fontsize=16)
            if save_plots:
                plt.savefig(fname+"-"+argu+"."+exten, dpi = 75)
    # Computing the beta star plot
    bearray = array(simuls['belist'][0])
    bearray.sort()
    surplus = U(Ef(kstar) - (r + d)*kstar)/(1 - bearray) -  U(f(kstar, zH)) - \
              bearray * (U(f(0, zH)) * pH + U(f(0, zL)) * pL)/(1 - bearray)
    plt.figure()
    plt.plot(bearray, surplus, 'o-')
    plt.plot(bearray, zeros(size(bearray)), '-')
    plt.xlabel(r'$\beta$')
    if titles:
        plt.title(r'Surplus and $\beta^\star$', fontsize=18)
    plt.savefig("plot-betastar."+exten, dpi = 75)

# ------------------ Basic Parameters --------------------------
r = .05                 # international interest rate
a = .33                 # share of capital
a0 = 1                  # another production function parameter
d = .05                 # depreciation rate
rho = 2                 # power utility parameter
sim_periods = 2000000   # simulation periods
# structure of the shocks
zH = 1.0
zL = .9
pH = .5
# some more parameters.
pL = 1 - pH
Ez = zH * pH + zL * pL  # mean shock
R = 1 + r
parameters = {'r': r, 'a': a, 'a0': a0, 'd': d,
              'rho': rho, 'zH': zH, 'zL': zL, 'pH': pH,
              'sim_periods': sim_periods}
bigT = time()
begrid = linspace(.1, 1/R - .02, 20) # discount factor grid
simulations = {}                     # a dictionary to store simulations
kstar = optimize.fmin(lambda k: - pH * f(k, zH) - pL * f(k, zL) +
                      (r + d) * k, 0., disp = 0)[0] 
for be in begrid:
    # Run simulation over different betas.
    print '\n------------------------------------------------'
    print 'Iterating over beta: ', be
    t = time()
    bR = be * R
    A0 = (pH * U(f(0, zH)) + pL * U(f(0, zL))) / (1 - be)
    vmin = A0
    vmax = U(f(kstar, zH)) + be * A0
    B0 = compute_first_best()
    Bv, pol = do_main_iter(B0, do_plots=False, stop_iter=10**(-4),
                           fast_iters=10)
    simulations[be] = {'Bv': Bv, 'pol': pol}
    print '-> iteration time: ', (time()-t) / 60, 'minutes'
# io.save('simulation_data', {'simulations':simulations})
# to load: from simulation_data import simulations, simulstats, parameters
#          r, a, a0, d, rho, zH, zL, pH = parameters
print 'simulating model and computing statistics'
simulstats = simulate_stats(simulations, path_length=10 ** 6) # simulates
# save the data output
# io.save('simulation_data', {'parameters': parameters,
#                             'simulations': simulations,
#                             'simulstats': simulstats})
do_stats_plots(simulstats, save_plots=False, exten="pdf")    # do the plots
plt.show()
print 'Total time:', round((time() - bigT) / 60, 2), 'minutes'


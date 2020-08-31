###################################
# Virus-IFN model parameter fitting
# Emily Ackerman 
# August 2020
###################################

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from lmfit.model import save_modelresult
from lmfit.printfuncs import *
from scipy.integrate import odeint
import corner 

tvec = np.genfromtxt('data_h1n1.csv', usecols = 0, delimiter = ',', skip_header = 1) #timepoints
data = np.genfromtxt('data_h1n1.csv', usecols = [1,2], delimiter = ',', skip_header = 1) #data, **usecols must change when changing model states

# 2 state ODE model 
def f2(u, t, p):
    try: 
        k = p['k'].value
        K = p['K'].value
        r_ifn_v = p['r_ifn_v'].value        
        d_v = p['d_v'].value
        p_v_ifn = p['p_v_ifn'].value
        d_ifn = p['d_ifn'].value
    except: 
        k, K, r_ifn_v, d_v, p_v_ifn, d_ifn = p
    V, IFN = u 
    V0 = 6.382
    IFN0 = 0.26
    dv = k*V*(1-V/K) - r_ifn_v*(IFN-IFN0)*V - d_v*V
    difn = p_v_ifn*V - d_ifn*(IFN - IFN0)
    return [dv, difn]

#initial conditions for state, time span 
u0 = [1.66,0.0]
tspan = (0., 100.)

# integrator function
def g(t, u0, p):
    sol = odeint(f2, u0, t, args=(p,))
    return sol

# residual error 
def residual(p, t, data):
    u0 = [6.382, 0.26]
    model = g(t, u0, p)
    err = np.square(model - data) # sum of squared errors
    err = err[~np.isnan(err)]
    err = (err/np.size(err)).ravel()
    target_length = np.shape(data)[0]*np.shape(data)[1]
    if np.size(err) != target_length: 
        err = np.append(err,np.zeros(target_length - np.size(err)))
    return err 

# set initial parameters with bounds 
params = Parameters() 
params.add('k', value= 786.46, min=0.00001, max = 1000)
params.add('K', value= 506, min=0.00001, max=1000)
params.add('r_ifn_v', value= 116.049346624, min=0.00001, max=1000)
params.add('d_v', value= 203.90836, min=0.00001, max=1000)
params.add('p_v_ifn', value= 298.2702, min=0.00001, max=1000)
params.add('d_ifn', value= 688.2520136356, min=0.00001, max=1000)


# fit model and find predicted values
result = minimize(residual, params, args=(tvec, data), method='emcee', steps = 50000, progress = True, nan_policy = "omit")
final = data + result.residual.reshape(data.shape)

# Plot model fit against data 
fig, axs = plt.subplots(2)
axs[0].plot(tvec, data[:,0], 'o')
axs[0].plot(tvec, final[:,0], '--')
axs[0].set_title('Virus')
axs[0].set_ylabel('log10(PFU/mg)')

axs[1].plot(tvec, data[:,1], 'o')
axs[1].plot(tvec, final[:,1], '--')
axs[1].set_title('IFN')
axs[1].set_xlabel('days')
axs[1].set_ylabel('Gene expression')

plt.savefig('fits.pdf')

# display fitted statistics
report_fit(result)

# plot acceptance
plt.figure(2)
plt.plot(result.acceptance_fraction)
plt.xlabel('walker')
plt.ylabel('acceptance fraction')
plt.savefig('acceptance.pdf')

# create corner plot of parameter distributions 
emcee_plot = corner.corner(result.flatchain, labels=result.var_names,
                           truths=list(result.params.valuesdict().values()))
plt.savefig('corner.pdf')

highest_prob = np.argmax(result.lnprob)
hp_loc = np.unravel_index(highest_prob, result.lnprob.shape)
mle_soln = result.chain[hp_loc]
for i, par in enumerate(params):
    params[par].value = mle_soln[i]


print('\nMaximum Likelihood Estimation from emcee       ')
print('-------------------------------------------------')
print('Parameter  MLE Value   Median Value   Uncertainty')
fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
for name, param in params.items():
    print(fmt(name, param.value, result.params[name].value,
              result.params[name].stderr))


#save model result
#save_modelresult(result, '2_state_result_8_12_20.sav')

import numpy as np
import emcee
import scipy.optimize as op
import matplotlib.pyplot as plt
import corner
import os
import scipy.stats as sstats

# local imports
import beer

def lnlike(theta, model, x, y, yerr):
    """ The log likelihood function.
    """
    return -np.nansum(0.5 * np.log([2 * np.pi] * len(y)))\
           -np.nansum(np.log(yerr))\
           -0.5*np.nansum(((y-model(x, *theta))/yerr)**2) 

def lnprior(theta, priors):
    """ Set prior boundaries.
    """
    nparam = len(theta)

    pass_count = 0
    for i in range(nparam):
        if priors[i][0] < theta[i] < priors[i][1]:
            pass_count += 1

    if pass_count == nparam:
        return 0.0
    else:
        return -np.inf

def lnprob(theta, model, priors, x, y, yerr):
    """ Returns the log likelihood function if the walker stays within 
    the priors.
    """
    lp = lnprior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, model, x, y, yerr)

def get_theta(guess_list, *args):
    """ Returns the parameter space theta.

    guess_list: list of initial parameter guesses
    args: arguments from lnlike() (model, x, y, y_errs)
    """
    nll = lambda *args : -lnlike(*args)
    result = op.minimize(nll, guess_list, args=args)
    theta = result["x"]
    return theta

def get_samples(ID, params, pos, ndim, nwalkers, nsteps, theta, \
    model, priors, x, y, yerr, convergenceplot_name=None, cornerplot_name=None):
    """ Returns the samples from emcee.
    """
    # pos = [np.array([theta[0] + 1e-4*np.random.randn(), theta[1] - 0.1 * np.random.randn()]) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
        args=(model, priors, x, y, yerr))
    sampler.run_mcmc(pos, nsteps)

    # if you want to check your burn in:
    if convergenceplot_name != None:
        if ndim==1:
            fig = plt.figure()
            plt.plot(sampler.chain[:,:,0].T, color='black', alpha=0.1)
            plt.ylabel(params[0])
            plt.xlabel('steps')
        else:
            fig, axs = plt.subplots(ndim, 1, sharex=True)
            for i in range(ndim):
                axs[i].plot(sampler.chain[:,:,i].T, color='black', alpha=0.1)
                axs[i].set_ylabel(params[i])
            axs[ndim-1].set_xlabel('steps')

        fig.set_size_inches(10, 10)
        plt.savefig(os.getcwd() + '/targets/' + str(ID) + '/' + convergenceplot_name)

    samples = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim))

    def plot_corn():
        labels = params
        quantiles=[0.16, 0.5, 0.84]
        fig = corner.corner(samples, labels=labels, quantiles=quantiles, \
            show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig(os.getcwd() + '/targets/' + str(ID) + '/' + cornerplot_name)

    if cornerplot_name != None:
        plot_corn()

    return samples






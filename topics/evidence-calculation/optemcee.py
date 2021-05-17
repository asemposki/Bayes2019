import numpy as np
from tqdm import tqdm

def run_mcmc(sampler, p0, nsteps, thin=1):
    '''
    Provided an instance of Sampler (sampler), an nwalkers by ndim initial
    position array (p0), and nsteps, this function returns the chain,
    loglikelihood (logl), and logprior values.
    '''
    ntemps = np.size(sampler.betas)
    chain = np.zeros((nsteps, ntemps, sampler.nwalkers, sampler.ndim))

    pbar = tqdm(total=nsteps)
    index = 0
    logl = np.zeros((nsteps, ntemps, sampler.nwalkers))
    logpi = np.zeros((nsteps, ntemps, sampler.nwalkers))
    for result in sampler.sample(p0, thin_by=thin):
        chain[index, :, :, :] = result.x
        logl[index, :, :] = result.logl
        logpi[index, :, :] = result.logP
        index += 1
        pbar.update(1)
        if index >= nsteps:
            break
    pbar.close()

    return chain, logl, logpi

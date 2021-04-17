import numpy as np


def metropolis_hastings(x_init, proposal, prior, log_likelihood, data,
                        proposal_kwargs=None, samples=10000, burn_in=0.0):
    """ Algorithm 27.3 Metropolis-Hastings MCMC sampling from
    David Barber's Bayesian Reasoning and Machine Learning.
    
    Modified to use log of acceptance ratio for numerical stability.

    :param x_init: numpy.ndarray or int, float
        The initial state.
    :param proposal: callable(x)
        The proposal distribution function.
    :param prior: callable(x)
        The prior distribution function.
    :param log_likelihood: callable(x, data)
        The likelihood function for x given the data. 
        Should be the log of the likelihood for numerical 
        stability.
    :param data: array_like
        Data used to determine the likelihood of parameters.
    :param proposal_kwargs: dict
        Keyword arguments for proposal function. Default is None.
    :param samples: int
        The number of samples to take. Default is 10000.
    :param burn_in: float
        The percentage of samples from the beginning to 
        remove. Must be a float between 0 and 1. 
        0 is equivalent to removing no samples and 1 is equivalent
        to removing them all.
    :return: numpy.ndarray, numpy.ndarray
        The samples taken. The first array is the accepted samples and the
            second is the rejected samples.
        If x_init is a int or float a 1 dimensional arrays will be returned.
        If x_init is a numpy.ndarray or list a 2 dimensional arrays will be
            returned. The fist dimension contains the samples, and the
            second each element of all the sampled states.
    """

    if proposal_kwargs is None:
        proposal_kwargs = dict()

    rejected = []
    accepted = []
    
    if samples < 1:
        
        raise Exception('samples must be greater than 0')
        
    if burn_in < 0 or burn_in > 1:
        
        raise Exception('burn_in must be between 0 and 1')
    
    x = x_init  # set x0
    
    burn_in_idx = int(samples * burn_in)

    for i in range(1, samples):

        # generate a random candidate state with the given the proposal function
        x_cand = proposal(x, **proposal_kwargs)
        
        # compute acceptance ratio
        a = log_likelihood(x_cand, data) + np.log(prior(x_cand)) - log_likelihood(x, data) - np.log(prior(x))

        if a >= 0:  # 0 instead of 1 because we took the log of the acceptance ratio

            x = x_cand
            if i >= burn_in_idx:
                accepted.append(x_cand)

        else:

            u = np.random.uniform(0, 1)  # draw a random value u uniformly from the unit interval [0, 1]

            if u < np.exp(a):  # need to convert back to probability (vs log probability)

                x = x_cand
                if i >= burn_in_idx:
                    accepted.append(x_cand)

            else:
                
                if i >= burn_in_idx:
                    rejected.append(x_cand)
    
    # convert to numpy and remove burn_in
    accepted = np.array(accepted)
    rejected = np.array(rejected)

    return accepted, rejected  # return samples


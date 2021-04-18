import numpy as np


def metropolis_hastings(x_init, proposal, log_prior, log_likelihood, data,
                        proposal_kwargs=None, samples=10000, burn_in=0.0):
    """ Algorithm 27.3 Metropolis-Hastings MCMC sampling from
    David Barber's Bayesian Reasoning and Machine Learning.
    
    Modified to use log of acceptance ratio for numerical stability.

    :param x_init: The initial state. If a numpy.ndarray is passed it must be one dimensional.
    :type x_init: numpy.ndarray, int, or float
    :param proposal: The proposal distribution function.
    :type proposal: callable(x)
    :param log_prior: The prior distribution function for x. Should be the log of the prior for numerical stability.
    :type log_prior: callable(x)
    :param log_likelihood: The likelihood function for x given the data. Should be the log of the
        likelihood for numerical stability.
    :type log_likelihood: callable(x, data)
    :param data: Data used to determine the likelihood of parameters.
    :type data: numpy.ndarray
    :param proposal_kwargs: Keyword arguments for proposal function. Default is None.
    :type proposal_kwargs: dict, optional
    :param samples: The number of samples to take. Default is 10000.
    :type samples: int, optional
    :param burn_in: The percentage of samples from the beginning to remove. Must be a float between 0 and 1.
        0 is equivalent to removing no samples and 1 is equivalent to removing them all. Default is 0.
    :type burn_in: float, optional
    :return: The samples taken. The first array is the accepted samples and the second is the rejected samples.
        If x_init is a int or float a 1 dimensional arrays will be returned. If x_init is a numpy.ndarray or list a
        2 dimensional arrays will be returned. The fist dimension contains the samples, and the second
        each element of all the sampled states.
    :rtype: numpy.ndarray, numpy.ndarray
    """

    def posterior(beta):
        """ Computes the posterior of a given parameter beta.

        :param beta: The parameter to compute the posterior for.
        :type beta: numpy.ndarray, int, or float
        :return: Returns the calculated posterior. Type depends on beta.
        :rtype: numpy.ndarray, int, or float
        """
        return log_likelihood(beta, data) + log_prior(beta)
    
    if isinstance(x_init, np.ndarray):

        if x_init.ndim > 1:
            raise Exception('x_init must be a one dimensional array')
    
    if samples < 1:
        
        raise Exception('samples must be greater than 0')
        
    if burn_in < 0 or burn_in > 1:
        
        raise Exception('burn_in must be between 0 and 1')
    
    if proposal_kwargs is None:
        proposal_kwargs = dict()
    
    rejected = []
    accepted = []
    
    x = x_init
    
    burn_in_idx = int(samples * burn_in)

    for i in range(1, samples):

        # generate a random candidate state with the given proposal function
        x_candidate = proposal(x, **proposal_kwargs)
        
        # compute acceptance ratio
        a = np.exp(posterior(x_candidate) - posterior(x))

        if a >= 1:

            x = x_candidate
            if i >= burn_in_idx:
                accepted.append(x_candidate)

        else:

            u = np.random.uniform(0, 1)  # draw a random value u uniformly from the unit interval [0, 1]

            if u < a:

                x = x_candidate
                if i >= burn_in_idx:
                    accepted.append(x_candidate)

            else:
                
                if i >= burn_in_idx:
                    rejected.append(x_candidate)
    
    # convert to numpy
    accepted = np.array(accepted)
    rejected = np.array(rejected)

    return accepted, rejected

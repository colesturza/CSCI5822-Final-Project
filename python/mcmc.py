import numpy as np


def metropolis_hastings(x_init, samples, proposal, acceptance):
    """ Algorithm 27.3 Metropolis-Hastings MCMC sampling from
    David Barber's Bayesian Reasoning and Machine Learning

    :param x_init: array-like or int, float
        The initial state.
    :param samples: int
        The number of samples to take.
    :param proposal: callable(x)
        The proposal distribution function.
    :param acceptance: callable(x, x_prime)
        The acceptance distribution function.
    :return: numpy.ndarray
        The samples taken.
        If x_init is a int or float a 1 dimensional array will be returned.
        If x_init is a numpy.ndarray or list a 2 dimensional array will be
            returned. The fist dimension contains the samples, and the
            second each element of all the sampled states.
    """

    # initialize array to store samples
    if isinstance(x_init, np.ndarray):

        if x_init.ndim > 1:
            raise Exception('x_init must be a one dimensional array')

        n = x_init.shape[0]

        x = np.array((samples, n), dtype=np.float64)

    elif isinstance(x_init, list):

        n = len(x_init)

        x = np.array((samples, n), dtype=np.float64)

    elif isinstance(x_init, (int, float)):

        x = np.array(samples, dtype=np.float64)

    else:

        raise Exception('x_init must be a numeric type, list, or numpy.ndarray')

    x[0] = x_init  # set x0

    for l in range(1, samples):

        x_cand = proposal(x[l - 1])  # generate a random candidate state with the given the proposal function

        a = acceptance(x[l - 1], x_cand)  # Calculate the acceptance probability

        if a >= 1:

            x[l] = x_cand

        else:

            u = np.random.uniform(0, 1)  # draw a random value u uniformly from the unit interval [0, 1]

            if u < a:

                x[l] = x_cand

            else:

                x[l] = x[l - 1]

    return x  # return samples

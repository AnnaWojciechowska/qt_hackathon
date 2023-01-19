"""Bulk vector data statistics

It is assumed that data has been read in by coincsv.read_dataset(),
one set for passed data and one for failed data, otherwise
corresponding to one another, using the columns returned by one when
reading the other, so that matching positions in vectors do correspond
to matching parameters of the VMs.

"""
import numpy, scipy

def mean_vary(data):
    """Returns mean and variance of data.

    Each entry in data should be a numpy.array(); all should have the
    same shape.  The mean is a vector of the same shape as each entry
    in data.  The variance is a square tensor with the shape of
    numpy.outer(mean, mean).  It is guaranteed to be positive-definite
    and symmetric (a.k.a. a metric).

    """
    m = numpy.average(data, 0)
    vary = sum(numpy.outer(v - m, v - m) for v in data) / len(data)
    return m, vary

def compare(passing, failed):
    """Report on the difference between passing and failing data.

    Returns (for now) just the difference between their means.
    """
    p, pv = mean_vary(passing)
    f, fv = mean_vary(failed)
    delta = f - p
    # TODO: analysis of relative variance

    return delta

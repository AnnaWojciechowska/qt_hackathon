"""Bulk vector data statistics

It is assumed that data has been read in by coincsv.read_dataset(),
one set for passed data and one for failed data, otherwise
corresponding to one another, using the columns returned by one when
reading the other, so that matching positions in vectors do correspond
to matching parameters of the VMs.

"""
import numpy, scipy
from coincsv import read_subs

def eigenbasis(form):
    vals, vecst = numpy.linalg.eig(form)
    return vals, vecst.T

def dualbasis(basis):
    return numpy.linalg.inv(basis).T

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

    Returns a tuple whose first entrance is the difference between the
    means of the datasets. Each subsequent entry is a pair (f, p) of
    corresponding data for the failed and passing datasets; first the
    eigenvalues of the two datasets, then their eigenbases.  Each
    eigenbasis is orthonormal for the standard metric; each list of
    eigenvalues is in decreasing order, with order corresponding to
    that of the matching eigenbasis.

    """
    p, pv = mean_vary(passing)
    f, fv = mean_vary(failed)
    delta = f - p

    pscale, pbasis = eigenbasis(pv)
    fscale, fbasis = eigenbasis(fv)
    # Some eigenvalues come out negative: but (now that we're
    # rescaling all columns to have O(1) median) they're tiny, so use
    # them to get the tolerance for how close to zero counts as a zero
    # eigenvalue.  It looks like 1e-12 is a fair cut-off.
    assert all(all(x > -1e-12 for x in s) for s in (pscale, fscale))
    # Given 25 columns with non-zero entries, a first sample shows 5
    # or 6 zero eigenvalues, boosted to 13 or 14 when we treat those
    # less than 1e-12 as zero.

    # TODO: work out how to mutually-diagonalize pv and fv.

    assert len(pscale) == len(fscale)
    return delta, (fscale, pscale), (fbasis, pbasis)

def _to_nearest_int(value):
    mid, bit = divmod(value, 1)
    mid = int(mid) # it already was a whole number, but as a real
    if bit > 0.5:
        return mid + 1
    if bit < 0.5:
        return mid
    # IEEE rounding half to even:
    return mid + 1 if mid & 1 else mid

def _asratio(value):
    """Approximates value by a ratio of whole numbers.

    If values is bigger than ten, this is the nearest whole number; if
    less than a tenth, then the inverse of the nearest whole number to
    its inverse is used; otherwise a tolerable approximation is found
    using smallish numbers.  The resulting p/q is returned as a twople
    (p, q).

    """
    # TODO: for value >> 10 or << 0.1, return the nearest power of ten
    # and teach rescale to represent it suitably in revised column
    # names.
    assert value > 0
    if value == int(value):
        return int(value), 1
    if value < 0.1:
        return 1, _to_nearest_int(1 / value)
    if value > 10:
        return _to_nearest_int(value), 1

    rough = ((_to_nearest_int(i * value), i) for i in range(1, 51))
    rough = sorted((value - p / q, p, q) for p, q in rough)
    err, p, q = rough[0]
    return p, q

def medial(arrays):
    """Returns a rational approximation to the median of each component.

    Each entry in arrays is assumed to be a numpy.array() of some
    given size, with different components having potentially very
    different scales.  For each component, this looks at that
    component of each entry in arrays: if all are zero, it yields
    None.  Otherwise, it approximates the median of the absolute
    values as a ratio p/q of whole numbers and yields a tuple (p, q).

    """
    for vs in zip(*arrays):
        vs = sorted(abs(v) for v in vs if v)
        if not vs:
            yield None
        else:
            mid, bit = divmod(len(vs), 2)
            median = vs[mid] if bit else (vs[mid - 1] + vs[mid]) * .5
            yield _asratio(median)

def rescale(scales, cols, *data):
    cols = ((s[1], c, s[0]) for s, c in zip(scales, cols) if s is not None)
    yield tuple(c if p == 1 == q else f'{p}*{c}' if q == 1
                else f'{c}/{q}' if p == 1 else f'{p}*{c}/{q}' for p, c, q in cols)
    for block in data:
        yield tuple(numpy.array([s[1] * v / s[0]
                                 for s, v in zip(scales, vs) if s is not None])
                    for vs in block)

def contrast(pairdir):
    """Contrast failed/ with passed/ datasets under pairdir.

    Singe parameter, pairdir, should name a directory under which may
    be found data, in the CSV-vased form expected by coincsv, in each
    of two subdirectories failed/ and passed/.

    This attempts to report on the salient differences between the
    two.

    """
    cols, passed, failed = read_subs(pairdir, 'passed', 'failed')
    assert all(len(v) == len(cols) for v in passed + failed)
    # The scales of different fields are wildly incompatible, so
    # rescale each component (by a rational) to make its abs's median
    # O(1).  In the process, filter out any columns that are all-zero
    # and change the column label of each survivor to reflect the
    # rescaling.
    scales = tuple(medial(passed + failed))
    assert len(scales) == len(cols)
    cols, passed, failed = rescale(scales, cols, passed, failed)
    delta, scales, bases = compare(passed, failed)
    return cols, delta, scales, bases

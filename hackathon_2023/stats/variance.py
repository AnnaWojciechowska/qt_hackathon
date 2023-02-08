"""Bulk vector data statistics

It is assumed that data has been read in by coincsv.read_dataset(),
one set for passed data and one for failed data, otherwise
corresponding to one another, using the columns returned by one when
reading the other, so that matching positions in vectors do correspond
to matching parameters of the VMs.

"""
import numpy
from coincsv import read_subs
from math import log10

class Space (object):
    """Description of a space or some subspace of one.
    """
    def __init__(self, components, coords=None):
        """Sets up a space or a subpace of it.

        Each entry in components names a component of the vectors in
        the underlying vector space.  If coords is (None, its default,
        or) not given, the whole space is described, using the
        coordinates thus named; its dimension is len(components).
        Otherwise, each entry in coords should be a covector of the
        underlying space that determines a component in the
        co-ordinates to be used for the subspace to be described; the
        subspace's dimension is len(coords).
        """
        self.__axes = components
        if coords is not None:
            coords = numpy.array(tuple(coords))
            assert all(len(x) == len(components) for x in coords)
        self.__coords = coords

    def entry(self, vector, axis):
        """Extract an entry from a vector by name.

        Takes a vector in the underlying space's co-ordinates and the
        name of one of those co-ordinates; returns the vector's entry
        corresponding to that co-ordinate.  Raises ValueError if the
        given name does not appear in self's list of co-ordinate
        names.
        """
        return vector[self.__axes.index(axis)]

    def coordinates(self, vector):
        """Map from underlying space vector to subspace co-ordinates.
        """
        if self.__coords is None:
            return vector
        return self.__coords.dot(vector)

    def composed(self, covector):
        """Map from subspace covector to underlying space covector.
        """
        if self.__coords is None:
            return covector
        return covector.dot(self.__coords)

    def describe(self, covector):
        return self.__describe(self.composed(covector))

    def __describe(self, covector):
        text = plus = space = ''
        for name, scale in zip(self.__axes, covector):
            scale = _simplify(scale)
            if not scale:
                continue
            if scale == -1:
                text += f'{space}-{name}'
            elif scale == 1:
                text += f'{space}{plus}{name}'
            elif scale < 0:
                text += f'{space}{scale:g} * {name}'
            else:
                text += f'{space}{plus}{scale:g} * {name}'
            plus, space = '+', ' '
        return text

    @staticmethod
    def __as_direction(covector):
        """Returns a maybe-more-printable covector parallel to the given one.

        Rescales to give one of the moderately small but not tiny
        entries magnitude 1, and with at least as many of the non-tiny
        entries positive as negative."""
        scales = sorted(abs(v) for v in covector if abs(v) > 1e-12)
        if scales:
            # Rescale by the smallest within a factor of 1000 of the biggest:
            covector /= min(s for s in scales if s > scales[-1] / 1e3)
        neg = len([x for x in covector if x < -1e-12])
        pos = len([x for x in covector if x > 1e-12])
        if neg > pos:
            return -covector
        return covector

    def rescaled(self, scales):
        """Return a (sub)space based on self, rescaling each coordinate.

        The iterable scales is iterated once; it must have as many
        entries as self's coordinates() returns.  If any entries in it
        are None, the corresponding coordinates of self are omitted
        from the resulting space.  Each coordinate of the resulting
        space is a coordinate of self divided by the corresponding
        scale.  The returned Space has an attribute .ignored which is
        a tuple of descriptions of ignored directions in the
        underlying space."""
        scales = tuple(scales)
        assert len(scales) == len(self.__axes)
        units = self.__coords
        if units is None:
            units = numpy.identity(len(self.__axes))
        assert all(len(x) == len(self.__axes) for x in units)
        res = Space(self.__axes, (v / 10 ** s for s, v in zip(scales, units)
                                  if s is not None))
        res.ignored = tuple(self.__describe(self.__as_direction(u))
                            for s, u in zip(scales, units) if s is None)
        return res

    def __diagonalise(self, form):
        """Return a (sub)space based on self that makes form diagonal.

        Presumes that form is a sum of tensor-squares of vectors,
        i.e. a positive semi-definite quadratic form on covectors, for
        self's (sub)space (i.e. expressed in its coordinates).  Finds
        coordinates that diagonalise this form and uses those with
        non-negligible diagonal entries to generate a subspace of self
        in which the form's representation is unit-diagonal.  Sets the
        .ignored property of the result to a tuple of descriptions of
        directions in the null-space of the form."""
        scale, basis = _eigenbasis(form)
        assert min(scale) >= -1e-12, (scale, form)
        res = Space(self.__axes, (self.composed(b / s**.5)
                                  for s, b in zip(scale, basis) if s > 1e-12))
        return res, tuple(self.__as_direction(self.composed(b))
                          for s, b in zip(scale, basis) if s <= 1e-12)

    def parameterise(self, data):
        """Find the right parameters to describe the given data.

        The data are given in the co-ordinates of the underlying space.
        """
        mean, vary = _mean_vary(data)
        if self.__coords is not None:
            vary = self.__coords.dot(vary).dot(self.__coords.T)
        res, ignored = self.__diagonalise(vary)
        res.ignored = tuple(f'{self.__describe(b)} = {_simplify(b.dot(mean)):g}'
                            for b in ignored)
        return res

def _simplify(scalar, tol=1e-8):
    if abs(scalar) < 1e-12:
        return 0
    try:
        if abs(scalar.imag) < tol:
            scalar = scalar.real
    except AttributeError:
        pass

    whole = _to_nearest_int(scalar)
    if abs(whole - scalar) < tol:
        return whole

    return scalar

def _eigenbasis(form):
    vals, vecst = numpy.linalg.eig(form)
    return vals, vecst.T

def _dualbasis(basis):
    return numpy.linalg.inv(basis).T

def _mean_vary(data):
    """Returns mean and variance of data.

    Each entry in data should be a numpy.array(); all should have the
    same shape.  The mean is a vector of the same shape as each entry
    in data.  The variance is a square tensor with the shape of
    numpy.outer(mean, mean).  It is guaranteed to be positive-definite
    and symmetric (a.k.a. a metric).

    """
    data = tuple(data)
    m = numpy.average(data, 0)
    vary = sum(numpy.outer(v - m, v - m) for v in data) / len(data)
    return m, vary

def _to_nearest_int(value):
    mid, bit = divmod(value, 1)
    mid = int(mid) # it already was a whole number, but as a real
    if bit > 0.5:
        return mid + 1
    if bit < 0.5:
        return mid
    # IEEE rounding half to even:
    return mid + 1 if mid & 1 else mid

def _scale(arrays):
    """Identify the scale of variation of each array.

    Yields one value, for each array in arrays, indicating the scale of
    variation. For an array with no variation in value, this is None.
    Otherwise is is an n for which values in the array vary by of
    order 10 ** n."""
    for vs in zip(*arrays):
        vs = sorted(abs(v) for v in vs if v)
        if not vs:
            yield None
            continue
        wide = vs[-1] - vs[0]
        assert wide >= 0, vs
        if wide <= 0:
            yield None
            continue

        low = len(vs) // 3
        mid = vs[-low if low else -1] - vs[low]
        if mid > 0:
            wide = mid
        yield _to_nearest_int(log10(wide))

def contrast(pairdir):
    """Contrast failed/ with passed/ datasets under pairdir.

    Singe parameter, pairdir, should name a directory under which may
    be found data, in the CSV-vased form expected by coincsv, in each
    of two subdirectories failed/ and passed/.

    This attempts to report on the salient differences between the
    two."""
    cols, passed, failed = read_subs(pairdir, 'passed', 'failed')
    space, gather = Space(cols), passed + failed
    assert all(len(v) == len(cols) for v in gather)
    kernel = lambda s: '\n\t'.join(s.ignored)
    def lindep(s):
        return f'linear dependenc{"ies" if len(s.ignored) > 1 else "y"}'

    # The scales of different fields are wildly incompatible, so
    # rescale each component (by a power of ten) to make its abs's
    # variation's median O(1).  In the process, filter out any columns
    # that are constant and change the column label of each survivor
    # to reflect the rescaling.
    rescaled = space.rescaled(_scale(gather))
    if rescaled.ignored:
        print(f'Ignoring constant columns:\n\t{kernel(rescaled)}\n')

    shared = rescaled.parameterise(gather)
    if shared.ignored:
        print(f'Found {lindep(shared)}:\n\t{kernel(shared)}')

    passive, flawed = (shared.parameterise(rs) for rs in (passed, failed))
    if passive.ignored:
        print(f'Found pass-only {lindep(passive)}:\n\t{kernel(passive)}')
    if flawed.ignored:
        print(f'Found fail-only {lindep(flawed)}:\n\t{kernel(flawed)}')

    # ...
    return (passed, passive), (failed, flawed)

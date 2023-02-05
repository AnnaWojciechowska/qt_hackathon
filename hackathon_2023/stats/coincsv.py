"""Ingest Coin data from csv files.

This provides the tools to read, from several directories, various csv
files to form vectors with knowledge of their co-ordinate names.
"""
from os import walk
from os.path import join, splitext
from datetime import datetime, timedelta

def _parse_time(text):
    when = datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ')
    assert not when.microsecond
    # Time-stamps are at 10s intervals but sometimes a file is saved
    # at a moment slightly off the round number ...
    sec = when.second % 10
    if sec > 5:
        return when + timedelta(seconds = 10 - sec)
    if sec:
        return when - timedelta(seconds = sec)
    return when

class _Rows (object):
    def __init__(self):
        self.__data = {}

    def read(self, stream, stem):
        heads = tuple(col if col == 'time' else f'{stem}/{col}'
                      for col in stream.readline().strip().split(','))
        for line in stream:
            toks, bok = line.split(','), {}
            for k, v in zip(heads, toks):
                if k == 'time':
                    bok[k] = _parse_time(v.strip())
                    assert bok[k].second % 10 == 0
                else:
                    try:
                        bok[k] = float(v.strip())
                    except ValueError:
                        pass
            if 'time' not in bok:
                continue
            try:
                data = self.__data[bok['time']]
            except KeyError:
                data = self.__data[bok['time']] = {}
            # cpu.csv has a cpu-total line before later cpu0... lines
            # disk.csv puts C: before D:
            # Keep the first where the rest are duplicated.
            if not any(k in data for k in bok if k != 'time'):
                data.update(bok)

    def columns(self):
        if not self.__data:
            return ()
        seq = iter(self.__data.values())
        ks = set(k for k in next(seq) if k != 'time')
        for v in seq:
            ks = ks.intersection(k for k in v)
        return tuple(sorted(ks))

    def rows(self, columns):
        ts = sorted(self.__data.keys())
        for t in ts:
            data = self.__data[t]
            if all(k in data for k in columns):
                yield tuple(data[k] for k in columns)

class _Scanner (object):
    def __init__(self, topdir, columns=None):
        self.columns = columns
        self.__walker = walk(topdir)

    def ingest(self):
        rows = []
        for dirname, ignore, files in self.__walker:
            if not files:
                continue
            data = _Rows()
            for name in files:
                stem, ext = splitext(name)
                # TODO: diskio.csv has a challenging detail: the drive letter;
                # that probably wants corss-talk into disk.csv
                if ext != '.csv' or stem == 'diskio':
                    continue
                with open(join(dirname, name)) as fd:
                    data.read(fd, stem)
            if self.columns is None:
                self.columns = data.columns()
            if not self.columns:
                raise ValueError(f'No compatible data in {dirname}')
            rows.extend(data.rows(self.columns))
        return tuple(rows)

def read_dataset(topdir, columns=None):
    """Digest each subdir of topdir into a sequence of vectors.

    If optional argument columns is None or absent, all data are read
    and a choice of orderings of the columns is made in the course of
    doing so, using all columns found for all timestamps.  Otherwise,
    columns should list the names of the columns to be gathered,
    combining in file/column form (as each file has its own set of
    columns), with the file's .csv extension omitted.  Any columns
    present in files but absent from columns (when given) are ignored;
    if any timestamp lacks an expected column, it is omitted.

    Each file is expected to have a time column; data for each
    time-stamp present in those columns is combined across files to
    produce a vector. It is assumed that distinct sub-directories'
    files do not have timestamps in common; reading all files in a
    directory that contains any files will deliver all data about any
    individual timestamp present in any of those files.

    Only columns whose data are parsed as float are included. The
    time-stamp is thus not one of the columns of the result, although
    it is internally recognised in order to combine values.

    Returns a twople: a tuple of vectors and a tuple of column names.
    The latter identifies the successive components of each vector in
    the former.

    """
    scan = _Scanner(topdir, columns)
    rows = scan.ingest()
    # That may change scan.columns.
    return rows, scan.columns

def read_subs(root, *subs):
    """Digest the dataset in each of a given list of subdirs of root.

    First argument is a parent directory; all subsequent arguments
    must be names, within that directory, of subdirectories, each of
    which contains a dataset of the kind understood by read_dataset().

    Returns a tuple whose first element is the list of column names,
    each subsequent element being the data for those columns read from
    the corresponding entry in subs.

    """
    subs = iter(subs)
    sub = join(root, next(subs))
    rows, columns = read_dataset(sub)
    data = [rows]
    for sub in subs:
        rows, c = read_dataset(join(root, sub), columns)
        assert c == columns
        data.append(rows)
    return columns, *data

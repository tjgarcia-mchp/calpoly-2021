# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:19:26 2021

@author: toemo
"""
import itertools as it
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
from warnings import warn
from collections import OrderedDict
from glob import iglob
as_strided = np.lib.stride_tricks.as_strided

get_class = lambda fn: int(os.path.basename(fn).split('.', 1)[0].rsplit('-', 1)[1])

# Have groupby work like it does in SQL (no duplicate keys)
groupby = lambda v, k: it.groupby(sorted(v, key=k), key=k)

ichain = it.chain.from_iterable

expandglobs = lambda filepatterns: ichain(iglob(ptrn) for ptrn in filepatterns)

def stackplot(ax, xlabels, legend, values, width=0.8):
    ax.set_prop_cycle(plt.cycler(color=plt.cm.Set3.colors))

    lastv = np.zeros(len(xlabels))
    for i, lk in enumerate(legend):
        ax.bar(xlabels, values[i], width, label=lk, bottom=lastv)
        lastv += values[i]
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=15)
    ax.legend(prop={'family': 'monospace'})

# Take M combined samples over all rows, keeping the samples as evenly
# distributed as possible
def sample_sets(x, M, kmax=None, key=None):
    L = len(x)
    if L == 0:
        return np.array([0], dtype=int)
    
    if key is None:
        key = len
    
    # Number of samples per row
    xn = np.array([*map(key, x)], dtype=int)

    # Number of samples in sorted order
    xnas = np.argsort(xn)

    # Array with number of elements to be chosen from each row
    m = np.zeros_like(xn)

    if kmax is not None:
        xn = np.minimum(xn, kmax)

    # Find the number of samples to choose per bin
    i = 0
    R = 0
    # N = M
    while i < L and M > 0:
        # Get the minimum number of samples contained in each remaining bin
        j = xnas[i]
        ni = xn[j]
        k = ni - m[i]

        # Get the number of samples to grab this loop
        n = min(M, k*(L - i))

        # Spread the samples over the remaining bins
        k, r = divmod(n, (L - i))

        # Add the k samples to each output bin
        m[i:] += k

        # Dole out remainder samples later, giving priority to the input order
        R += r
        
        # Update the sample counter
        M -= n
        
        # Skip ahead to the next largest bin
        i += 1
        while i < L:
            j = xnas[i]
            if xn[j] != ni:
                break
            i += 1

    # Unsort the array
    xnas = np.argsort(xnas)
    m = m[xnas]

    # Add the remainder to the first R bins wrt the input order
    mask = m < xn
    mask&= np.cumsum(mask) <= R
    m[mask] += 1
    # assert sum(m) == min(N, kmax*L if kmax else N)

    return m

def split_train_test(samples, test_portion, N=None, hashkey=None, groupkey=None, shuffle=True, rng=None, split_mode='discard', balance_groups=False, balance_hashes=False):
    '''
    Generate a train/test split
    Note: Assumes all elements with the same hashkey also have the same groupkey (but not vice-versa)
    N : scalar
        Maximum number of samples to choose
    hashkey : function
        allows you to pool data samples to disallow samples from similar distributions 
        (e.g. sub-segments from the same sample) splitting across folds
    groupkey : function
        allows you to specify sample groups which should be explicitly split 
        between folds (e.g. split all samples belonging to the same experiment across folds)
    split_mode : string
        How to handle sample groups that are too small to split
        'test' - place data into test split
        'train' - place data into training split
        'discard' - place data into discard
    '''
    test = []
    train = []
    discard = []
    
    assert split_mode in ('train', 'test', 'discard')

    if rng is None:
        rng = np.random
    
    # Split samples by hash code first
    if hashkey is None:
        hshmap = { i: [v] for i,v in enumerate(samples) }
    else:
        hshmap = { k: [*it] for (k, it) in groupby(samples, hashkey) }
    
    # Split hashed samples by attribute group
    if groupkey is None:
        # Put everything in the same group
        grpmap = { 0: [*hshmap] }
    else:
        grpmap = { k: [*it] for (k, it) in groupby(hshmap, groupkey) }
    
    # Shuffle to randomize the split across folds
    groups = [*grpmap]
    if shuffle:
        rng.shuffle(groups) 
    
    # Balance sampling according to the total number of samples belonging to a hash
    grplen = lambda gk: sum(len(hshmap[h]) for h in grpmap[gk])

    Ngrpmax = None
    if balance_groups:
        Ngrpmax = min(map(grplen, groups))
    
    if N is None:
        Nc = len(samples)
    else:
        Nc = min(N, len(samples))
    
    # Distribute each 'group' of samples between train and test folds
    for gk, m in zip(groups, sample_sets(groups, Nc, kmax=Ngrpmax, key=grplen)):
        # Select a portion for training, and rest is reserved for testing
        m_train = max(1, min(m-1, int((1-test_portion)*m + 0.5)))
        m_test = m - m_train
        hashes = grpmap[gk]
        try:
            assert (0 < m_train < m), "Not enough samples in group '%s' to allow splitting; samples assigned to %s fold" % (gk, split_mode)
            assert (len(hashes) > 1), "Not enough hashed samples in group '%s' to allow splitting; samples assigned to %s fold" % (gk, split_mode)
        except AssertionError as err:
            # if split_mode == 'raise':
            #     raise
            # warn(str(err))
            if split_mode == 'discard':
                m_train = m_test = 0                    
            elif split_mode == 'train':
                m_train = m
                m_test = 0
            elif split_mode == 'test':
                m_test = m
                m_train = 0
        
        # Shuffle to randomize the split across folds
        if shuffle:
            rng.shuffle(hashes)

        grphshmap = { h: hshmap[h] for h in hashes }
        hshlen = lambda h: len(hshmap[h])

        Nhshmax = None
        if balance_hashes:
            Nhshmax = min(map(hshlen, hashes))

        # Deal out samples
        # Since hash groups are not necessarily of the same size, alternate
        # between folds to keep things as balanced as possible
        f = 0            
        folds = [(train, m_train), (test, m_test)]
        fld, m_f = folds[f]
        # ndropped = 0
        for hk, m_h in zip(hashes, sample_sets(hashes, m, kmax=Nhshmax, key=hshlen)):
            hshsamples = grphshmap.pop(hk)
            if shuffle:
                rng.shuffle(hshsamples)
            
            fld += hshsamples[:m_h]
            m_f -= m_h
            # ndropped += len(hshsamples) - m_h

            if folds[f^1][1] > 0:
                # Alternate to other fold
                folds[f] = (fld, m_f)
                f ^= 1
                fld, m_f = folds[f]
            elif m_f == 0:
                break
            
            # Discard remaining samples in this hash group to maintain
            # the train/test ratio
            # discard += hshsamples[m_h:]

        # Any remaining hashes in this group which would throw off the
        # train/test ratio are discarded
        for v in grphshmap.values():
            # ndropped += len(v)
            discard += v
    
        # if ndropped:
        #     print("Dropped %d samples (%s)" % (ndropped, gk))

    return train, test, discard

def reduceruns(runs, mergethreshold=0, dropthreshold=0):
    if len(runs) == 0:
        return runs
    
    assert runs.shape[1] == 2
    
    if mergethreshold:
      segs = [runs[0]]
      for i in range(1, len(runs)):
          if runs[i,0] - segs[-1][1] - 1 < mergethreshold:
              #print("Segment %d merged into segment %d" % (i, len(segs)-1))
              segs[-1][1] = runs[i,1]
          else:
              segs.append(runs[i])
      runs = np.stack(segs)
  
    if dropthreshold:
      idx = (runs[:,1] - runs[:,0]) >= dropthreshold
      # if not idx.all(): 
      #     for i in np.nonzero(~idx)[0]:
      #         print("dropped segment %d (%d < %d)" % (i, runs[i,1] - runs[i,0], dropthreshold))
      runs = runs[idx]

    return runs

def mask2runs(mask, mergethreshold=0, dropthreshold=0):
  '''
  mask2runs
 
  Turns a boolean array into an array of indices indicating the start and end (technically end index + 1) indices of each run of 1's.
  '''
  runs = np.nonzero(mask[1:] ^ mask[:-1])[0] + 1
  #runs[1::2] -= 1 # Note that the prior step returns end indices as the end of a run plus 1
 
  # if the last run was a run of 1's, count the last epoch as the end of the run
  # similarly if the first bit started a run of 1's
  if mask[-1]: runs = np.r_[runs,len(mask)]
  if mask[0]: runs = np.r_[0,runs]
  
  runs = runs.reshape((-1,2))
 
  return reduceruns(runs, mergethreshold=mergethreshold, dropthreshold=dropthreshold)

def runs2mask(runs, shape=None, fill=0):
    if shape is None:
        shape = runs[-1,-1]
    x = np.zeros(shape, dtype=np.bool)
    for i in runs:
        x[slice(*i)] = 1
    
    x[runs[-1,-1]:] = fill
    
    return x

def maskruns(x, runs):
    return np.ma.masked_array(x, ~runs2mask(runs, x.shape))

def runs2slice(runs):
    runs.shape = (runs.size//2, 2)
    return np.r_[tuple(slice(*i) for i in runs)]

# Assumes blocks are split into evenly divided parts
# Accepts multi-dimensional input, but 1-d and 2-d are the only ones tested
class Strider(object):

    def __init__(self, blocksize, invhopratio=1):
        assert blocksize > 1, "blocksize of 1 not supported"
        assert (blocksize % invhopratio) == 0, "invhopratio must be a factor of blocksize"
        self.blocksize = blocksize
        self.hopsize = blocksize // invhopratio
        self.overlap = self.blocksize - self.hopsize

    def from_strided(self, blocks, dtype=None, shape=None, fill=0):
        nblocks = len(blocks)
        blockshape = blocks.shape[2:]
        #assert blocks.ndim > 1, "Blocked input should be at least 2-d"
        
        # import pdb; pdb.set_trace()
        
        # Assume that if the dimensions have been reduced, a function was applied across the windows
        # in which case from_strided will tile the function output to match the original input signal shape
        # FAILCASE: STFT.from_strided when NFFT > blocksize (i.e. numpy.fft is doing the padding)
        if blocks.ndim == 1:
            blocks = blocks.reshape((len(blocks), 1))
        elif blocks.shape[1] != self.blocksize and blocks.shape[1] != 1:
            blockshape = blocks.shape[1:]
            blocks = blocks.reshape(blocks.shape[:1] + (1,) + blocks.shape[1:])

        if dtype is None:
            dtype = blocks.dtype        
        if shape is None:
            shape = (nblocks * self.hopsize + self.overlap,) + blockshape
        elif np.prod(shape) < np.prod((nblocks * self.hopsize + self.overlap,) + blockshape):
            raise ValueError("shape=%s isn't large enough to hold output of shape %s" % (shape, (nblocks * self.hopsize + self.overlap,) + blockshape))
        
        if blocks.shape[1] == 1:
            """
            This is a trick reserved for reshaping the output of block aggregate functions 
            to match the original input signal shape by tiling (i.e. repeating) the function
            output; mostly for visualization purposes
            Example:
              strdr = Strider(200, 2)
              wx = strdr.to_strided(x)
              wxdB = 10 * np.log10(np.mean(x**2, axis=1, keepdims=True))
              xdB = strdr.from_strided(wxdB, shape=wx.shape)
            """
            array = np.ones(shape, dtype=dtype)
            subarry = array[:nblocks*self.hopsize]
            subarry.shape = (nblocks, self.hopsize) + blockshape
            subarry *= blocks[:nblocks] # Broadcast multiply   
            array[nblocks*self.hopsize:] *= fill
        elif self.overlap == 0 and np.prod(shape) == blocks.size:
            # Just collapse the second dimension back into the first
            array = blocks
            array.shape = (array.shape[0]*array.shape[1],) + blockshape            
        else:
            # Make a new array, copying out only the non-overlapping data
            array = np.ones(shape, dtype=dtype)
            array[:nblocks*self.hopsize] = blocks[:nblocks, :self.hopsize, ...].reshape((nblocks*self.hopsize,) + blockshape)
            array[nblocks*self.hopsize:nblocks*self.hopsize+self.overlap] = blocks[nblocks-1, self.hopsize:, ...].reshape((self.overlap,) + blockshape)
            array[nblocks*self.hopsize+self.overlap:] *= fill
        
        return array

    def to_strided(self, x, pad=False, fill=0):
        '''\
        Transforms input signal into a tensor of strided (possibly overlapping) segments
        
        Parameters
        ----------
        x : ndarray
            input array.
        pad : bool, optional
            Whether to pad the input x so that no samples are dropped. The default is False. !NB! This requires a copy of the input array to be made.

        Returns
        -------
        blocks : ndarray
            Strided array.

        '''
        writeable = (self.overlap == 0) # Only allow writing for non-overlapping strides
        
        blockshape = x.shape[1:]
        blockstrides = x.strides
        elemsize = int(np.prod(blockshape)) or 1
        
        nblocks, rem = divmod(x.size - self.overlap*elemsize, self.hopsize*elemsize)
        if nblocks < 0:
            nblocks = 0
            rem = self.blocksize*elemsize - x.size
        if pad and rem > 0:
            pad = np.ones((self.blocksize - (rem//elemsize),) + blockshape, dtype=x.dtype) * fill
            x = np.concatenate((x, pad))
            nblocks += 1
            writeable = True # This is non-overlapping memory now, so allow writing
        
        blocks = as_strided(x, shape=(nblocks, self.blocksize) + blockshape, strides=(self.hopsize*blockstrides[0],) + blockstrides, writeable=writeable)

        return blocks
    
    def __repr__(self):
        return "%s(blocksize=%d, hopsize=%d)" % (self.__class__.__name__, self.blocksize, self.hopsize)

class RSTFTStrider(Strider):
    
    def __init__(self, window, invhopratio=2, nfft=None):
        '''RSTFTStrider        

        Parameters
        ----------
        window : ndarray or scalar
            (ndarray) Pre-fft window to apply 
            (scalar) Number of sample points to use per FFT. In this case no windowing will be applied before FFT.
        invhopratio : TYPE, optional
            DESCRIPTION. The default is 2.
        nfft : int, optional
            FFT size (should be >= window size to avoid truncation). The default (None) sets the FFT size equal to the window size.

        Returns
        -------
        None.

        '''
        if np.isscalar(window):
            window = np.ones(window)
        self.nfft = nfft or len(window)
        super(RSTFTStrider, self).__init__(len(window), invhopratio)
        self.window = window

    def to_stft(self, x, pad=False, fill=0):
        '''\
        Transform input signal into a tensor of strided (possibly overlapping) windowed 1-D DFTs
        '''
        window = self.window.reshape(self.window.shape + (1,) * len(x.shape[1:]))
        X = self.to_strided(x, pad=pad, fill=fill) * window
        return np.fft.rfft(X, n=self.nfft, axis=1)
    
    def to_lfe(self, x, pad=False, fill=0, eps=1e-12):
        X = self.to_stft(x, pad=pad, fill=fill)
        return np.log(1 / self.nfft * np.abs(X)**2 + eps)

    def from_stft(self, X, dtype=float, shape=None):
        nblocks = len(X)
        blockshape = X.shape[2:]
        window = self.window.reshape(self.window.shape + (1,) * len(blockshape))
        #assert X.ndim > 1, "Blocked STFT input should be at least 2-d"
        
        if X.ndim == 1:
            X = X.reshape((len(X), 1))
        
        if shape is None:
            shape = (nblocks * self.hopsize + self.overlap,) + blockshape
        elif np.prod(shape) < np.prod((nblocks * self.hopsize + self.overlap,) + blockshape):
            raise ValueError("shape isn't large enough to hold output")
        
        x = np.zeros(shape, dtype=dtype)
        for i in range(nblocks):
            x[i*self.hopsize:i*self.hopsize+self.blocksize] += np.fft.irfft(X[i], n=self.nfft, axis=0)[:self.blocksize] * window
            
        return x * self.overlap / np.sum(self.window**2)
    
class SubprocessDict(object):
    def __init__(self, P, timeout=None):
        self.P = P
        self.dict = OrderedDict()
        self.timeout = timeout
    def __len__(self):
        return len(self.dict)
    def __iter__(self):
        return self.dict.__iter__()
    def iterpop(self):
        keys = list(self.dict.keys())
        for k in keys:
            yield self.pop(k)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.wait(self.timeout)
    def __setitem__(self, k, p):
        assert k not in self.dict, "Non-unique key '%s'" % k
        while sum(p.poll() is None for p in self.dict.values()) == self.P:
            pass
        self.dict[k] = p
    def __getitem__(self, k):
        return self.dict[k]
    def keys(self):
        return self.dict.keys()
    def values(self):
        return self.dict.values()
    def items(self):
        return self.dict.items()
    def pop(self, k=None):
        if k is None:
            keys = list(self.dict.keys())
            ki = 0
            while True:
                if self.dict[keys[ki]].poll() is None:
                    k = keys[ki]
                    break 
                ki = (ki + 1) % len(keys)
        p = self.dict.pop(k)
        if p.wait(self.timeout) != 0:
            raise subprocess.SubprocessError("Process returned exit code %d: %s" % (p.returncode, p.args))
        return k
    def clear(self):
        for k in self.dict:
            self.dict[k].kill()            
        self.dict.clear()
    def wait(self):
        for k in self.dict:
            self.dict[k].wait()
    def __repr__(self):
        return repr(self.keys())
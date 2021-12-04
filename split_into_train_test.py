# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:00:53 2021

@author: toemo
"""
#%%
import os, sys
import utils as ut
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import re

# Run script interactively or from the command line
isinteractive = plt.isinteractive()

# Set a seed for the RNG so results are reproducible
np.random.seed(10)

def cli(argv):
    from argparse import ArgumentParser

    parser = ArgumentParser(description='''\
    Split input files into stratified training and testing folds.
    ''')

    parser.add_argument('source_file', \
        help='JSON file defining target classes and where to source the files for each class.')
    parser.add_argument('dstdir', help='Output directory.')
    parser.add_argument('-r', '--test-ratio', default=0.2, type=float, \
        help='Desired test/train ratio. (%(default)s)')
    parser.add_argument('-b', '--balance-classes', '--balance', action='store_true', \
        help='''\
Limit the number of samples used per class to the class with the least number of samples; this keeps the number of samples per class balanced.'''
    )
    parser.add_argument('--balance-groups', action='store_true', \
        help='''\
Limit the number of samples used per attribute group to the attribute group with the least number of samples.'''
    )
    parser.add_argument('--balance-hashes', action='store_true', \
        help='''\
Limit the number of samples used per hash group to the hash group with the least number of samples.'''
    )
    parser.add_argument('--validation', action='store_true', \
        help='''\
Include a validation fold in addition to the train and test folds.'''
    )
    parser.add_argument('-m', '--max-samples', default=None, type=int, \
        help='Limit the number of samples used per class.')
    parser.add_argument('-c', '--copy', action='store_true', \
        help='Copy selected source files from input into output directory.')
    parser.add_argument('-P', '--no-plot', action='store_true', \
        help='Disable plotting.')   
    parser.add_argument('-S', '--no-stratify', action='store_true', \
        help='Disable stratified sampling.') 

    args = parser.parse_args(argv)

    return args

#%% Get input
if not isinteractive:
    args = cli(sys.argv[1:])
else:
    import shlex
    cmd = '--balance glassbreak//input.json test/'
    args = cli(shlex.split(os.path.expanduser(cmd)))

target_classes = json.load(open(args.source_file))

if not os.path.exists(args.dstdir):
    os.makedirs(args.dstdir)

#%% Build up mapping from classes to file names
class_files = { k: [*ut.expandglobs(v)] for k,v in target_classes.items() }

nsamples = { k: len(v) for k, v in class_files.items() }
print("Input samples per class:")
print(json.dumps(nsamples, indent=2))

# Drop empty classes
for k in list(class_files):
    if len(class_files[k]) == 0:
        class_files.pop(k)

#%% Split segments into train and test folds
# Input files should be organized into subdirectories like so:
#   <source-dir>/<source-dataset>_<source-category>/<filename>
# And files should be named as
#   <filename>[.pt<part-number>][.c<chapter-number>][.p<page-number>]
# where,
#   <part> is a subsample from a source file
#   <chapter> is a subsample from a part
#   <page> is a subsample from a chapter
# 
# The hashkey pools segments together that are just subsegments from a larger
# source file so that they are not allowed to split across folds
# This ensures that we don't evaluate performance based on samples that are too
# similar to what we trained with
strptrn = r'(?:\.p\d+)'   # < Don't split pages
# strptrn+= r'|(?:\.c\d+)'  # < Don't split chapters
# strptrn+= r'|(?:\.pt\d+)' # < Don't split parts
nohashptrn = re.compile(strptrn)
def hashkey(fn):
    srcdir = os.path.basename(os.path.dirname(fn))
    srcds, srccat = srcdir.split('_', 1)
    basename, ext = os.path.basename(fn).rsplit('.', 1)
    
    if srcds == 'ESC50':
        # We're careful here to hash ESC-50 samples by their original CLIP-ID
        # Note: ESC50 file names have format: {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
        hashedfn = basename.split('-')[1]
    elif srcds == 'DEMAND':
        # for DEMAND we do allow subsegments to split between folds
        hashedfn = basename
    else:
        # This ensures subsegments are *not* split across train/test folds
        hashedfn = nohashptrn.sub('', basename)
        
    # Ensure that category and hash are delimited by the first underscore ('_')
    srccat = srccat.replace('_','-')
    srcds = srcds.replace('_','-')

    return '_'.join((srcds, srccat, hashedfn))

# The groupkey pools segments together that belong to the same category (e.g.
# VacuumCleaner, DKITCHEN, etc.) so they can be distributed among both training
# and testing folds in the desired proportions
# Note: this function expects the hashkey as defined above as its input
def groupkey(hashkey):
    srcds, srccat = hashkey.split('_', 2)[:2]
    return '_'.join((srcds, srccat))

# Finally, make the split
kwargs = dict(
    split_mode='train', 
    N=args.max_samples,
    hashkey=hashkey,
    groupkey=None if args.no_stratify else groupkey,
    balance_groups = args.balance_groups,
    balance_hashes = args.balance_hashes,
    rng=np.random
)

train, test, discard = ut.split_train_test(class_files, args.test_ratio, **kwargs)

folds = dict(zip('train test'.split(), (train, test)))

if args.validation:
    # Split further into training and validation
    
    # Add back unused samples
    for k,v in discard.items():
        train[k].extend(v)

    train, val, discard = ut.split_train_test(train, args.test_ratio, **kwargs)
    
    folds['validation'] = val
    folds['train'] = train

# Balance out classes as needed
if args.balance_classes:
    for fld in folds.values():
        N = min(map(len, fld.values()))
        for cls, values in fld.items():
            fld[cls] = values[:N]

nsamples = { 
    fldk : { 
        k: len(v) for k, v in fldv.items() 
    } 
    for fldk, fldv in folds.items() 
}

print("Output samples per class:")
print(json.dumps(nsamples, indent=2))

#%% Generate data frame for exploration
keys = ('filename', 'dataset', 'hash', 'group', 'class', 'fold', 'duration')

rows = [\
    (os.path.basename(fn), os.path.dirname(fn).split('_', 1)[0], hashkey(fn), groupkey(hashkey(fn)), clsk, fldk, sf.info(fn).duration)
        for (fldk, fldv) in folds.items() 
        for (clsk, clsv) in fldv.items() 
        for fn in clsv
    ]

rows = pd.DataFrame(rows, columns=keys)

#%% Plot distribution of data by groups
lblfmt = '%s (%0.2f)'
lgndfmt = '%s (%0.2f//%dh%02dm)'
grpfmt = '%%-%ds'

classes = [*class_files.keys()]

groups = sorted(
    [*set(rows['group'])], 
    key=lambda k: sum(rows.query('group == @k').duration), reverse=True)

labels = [
    lblfmt % 
    (
        fk,
        sum(rows[rows['fold'] == fk].duration)/sum(rows.duration)
    ) for fk in folds.keys()
]

nsamples = np.array([
    [
        sum(rows[(rows['group'] == gk) & (rows['fold'] == fk)].duration)/60.0
        for fk in folds
    ] 
    for gk in groups
])

grpfmt = grpfmt % max(map(len, groups))
legend = [
    lgndfmt % 
    (
        grpfmt % gk, 
        sum(rows[rows['group'] == gk].duration)/sum(rows.duration), 
        *divmod(sum(rows[rows['group'] == gk].duration/60), 60)
    ) 
    for gk in groups
]

fig, ax = plt.subplots(figsize=(8,6))
ut.stackplot(ax, labels, legend, nsamples)
ax.set_ylabel('Duration (minutes)')
ax.set_title('Data sources distribution')
fig.tight_layout()

if not args.no_plot:
    plt.show()

fig.savefig(os.path.join(args.dstdir, 'fold_distribution.png'))

#%% Create listing files for each fold
import shutil

fold_abbrev = dict(
    zip(
        ['train', 'test', 'validation'],
        ['trn', 'tst', 'val']   
    )
)

# Write file paths out to text files
folds['discard'] = discard
for fldk, fldv in folds.items():
    listfn = os.path.join(args.dstdir, fldk + '_list.txt')

    subdir = os.path.join(args.dstdir, fldk)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    with open(listfn, 'w') as fh:
        for cls, files in fldv.items():
            for srcfn in files:                
                # Write paths relative to output directory
                if args.copy and fldk != 'discard':
                    bn = '.'.join((cls, fold_abbrev[fldk], os.path.basename(srcfn)))
                    dstfn = os.path.join(subdir, bn)
                    shutil.copyfile(srcfn, dstfn)
                    srcfn = dstfn
                
                dstfn = os.path.relpath(srcfn, args.dstdir).replace(os.path.sep, '/')
                print(dstfn, file=fh)

# -*- coding: utf-8 -*-
#%%
import os, sys, shutil
import subprocess
import soundfile as sf
import utils as ut
import numpy as np
from glob import iglob

# Max concurrent processes
P = 8

# Run script interactively or from the command line
isinteractive = not sys.__stdin__.isatty()

# Path to the sox binary
SOX='sox'

#%% Setup
def cli(argv):
    from argparse import ArgumentParser

    parser = ArgumentParser(description='''\
    Split input files into fixed length segments, dropping any silent segments.
    Output files will be named as <basename>.c<segment-number>p<subsegment-number>.<extension>
    ''')
    parser.add_argument('sourcefiles', nargs='+',
                        help='Input file name(s). Accepts multiple files and wildcard expressions.')
    parser.add_argument('dstdir',
                        help='Output directory.')
    parser.add_argument('-l', '--segment-length', default=(0,0), type=lambda s: tuple(map(float, s.split('-'))),
                        help='Segment length in seconds. Can be specified as a range delimited by as dash (i.e.<min_duration>-<max_duration>). If set to <=0, segments won\'t be split up. (%(default)s)')
    # parser.add_argument('-m', '--min-segment-length', default=None, type=float,
    #                     help='Minimum length of segments. (%(default)s)')
    parser.add_argument('-t', '--silence-threshold', default=-60, type=float,
                        help='Silence threshold in dBFS. If set to >=0, all silent segments will be included. (%(default)s)')
    parser.add_argument('-d', '--min-silence-duration', default=0.25, type=float,
                        help='Minimum duration (in seconds) signal level needs to drop below silence threshold to consider a segment as silence. (%(default)s)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity.')

    args = parser.parse_args(argv)
    args.silence_threshold = min(args.silence_threshold, 0)
    
    return args

if not isinteractive:
    args = cli(sys.argv[1:])
else:
    import shlex
    cmd = "-vv --segment-length=3-5 --silence-threshold=-60 --min-silence-duration=0.5 './MSSNSD/ShuttingDoor_*.wav' test/tmp"
    args = cli(shlex.split(os.path.expanduser(cmd)))

try:
    args.min_segment_length, args.segment_length = args.segment_length
except ValueError:
    args.min_segment_length = args.segment_length = args.segment_length[0]
sourcefiles = list(ut.expandglobs(args.sourcefiles))
# if os.path.exists(args.dstdir):
#     shutil.rmtree(args.dstdir)
# if not os.path.exists(os.path.join(args.dstdir, 'discard')):
#     os.makedirs(os.path.join(args.dstdir, 'discard'))
if not os.path.exists(args.dstdir):
    os.makedirs(args.dstdir)

#%% Split files by silence
# Manage the created subprocesses
procs = ut.SubprocessDict(P)

# Split files by silence; rename so that <basename>.<ext> becomes
# <basename>.c<segment-number>.<ext>
dstfiles = []
trim_silence = True
if args.silence_threshold != 0:
    # Split files by silence
    for srcfn in sourcefiles:
        bn, ext = os.path.basename(srcfn).rsplit('.', 1)
        dstfn = os.path.join(args.dstdir, bn + '.c%1n.tmp.' + ext)

        # This will split audio into segments whenever level falls below the
        # threshold for the specified time; it also trims anything below the
        # threshold from the file start.
        # (Note: with 'silence' filter a 't' suffix must be used to specify
        # duration in seconds; a bare number is assumed as number of samples)
        cmd = '"{0}" "{1}" "{2}" silence 1 1 {4}d 1 {3}t {4}d : newfile : restart'
        cmd = cmd.format(SOX, srcfn, dstfn, args.min_silence_duration, args.silence_threshold)
        if args.verbose:
            print(cmd)
        
        stdout = stderr = subprocess.DEVNULL if args.verbose < 2 else None
        procs[dstfn] = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)

    # Trim silence from both ends of files
    for srcfn in procs.iterpop():
        for segfn in iglob(srcfn.replace('.c%1n.tmp.', '.c*.tmp.')):
            with sf.SoundFile(segfn) as sfh:
                duration = len(sfh) / sfh.samplerate
            if duration < args.min_segment_length:
                os.remove(segfn)
                continue
            # input format should .<segnum>.tmp.<ext>
            dstfn = os.path.join(args.dstdir, os.path.basename(segfn).replace('.tmp.', '.'))
            dstfiles.append(dstfn)

            if not trim_silence:
                shutil.move(segfn, dstfn)
                continue
            
            # This will trim any signal at the start or end of the file that is
            # below the silence threshold
            cmd = '"{0}" "{1}" "{2}" silence 1 1 {4}d reverse silence 1 {3}t {4}d reverse'
            cmd = cmd.format(SOX, segfn, dstfn, args.min_silence_duration, args.silence_threshold)
            if args.verbose > 1:
                print(cmd)

            stdout = stderr = subprocess.DEVNULL if args.verbose < 2 else None
            procs[segfn] = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
else:
    # Simply copy the input files to the output directory
    for srcfn in sourcefiles:
        bn, ext = os.path.basename(srcfn).rsplit('.', 1)
        dstfn = os.path.join(args.dstdir, bn + '.c1.' + ext)
        shutil.copyfile(srcfn, dstfn)
        dstfiles.append(dstfn)

# Drop intermediate files
for srcfn in procs.iterpop():
    os.remove(srcfn)

# Collect the list of created files (should be formatted <source-file>.c[0-9]+.<ext>)
tmpfiles = list(ut.ichain((iglob(srcfn.replace('.c%1n.', '.c*.')) if '.c%1n.' in srcfn else [srcfn]) for srcfn in dstfiles))

# # Resample and apply a HPF
# for srcfn in tmpfiles:
#     dstfn = srcfn
#     cmd = 'python preprocess.py "{0}" "{1}"'.format(srcfn, dstfn)
#     procs[srcfn] = subprocess.Popen(cmd, shell=True)
#%% Split files into fixed length segments 
# Split files into segment_length segments, dropping files that are too short.
# Rename so that <basename>.c<segment-number>.<ext> becomes
# <basename>.c<segment-number>.p<sub-segment-number>.<ext>
filelens = []
dstfiles = []
for srcfn in tmpfiles:
    # Collect the duration of the silence split segments
    with sf.SoundFile(srcfn) as sfh:
        duration = len(sfh) / sfh.samplerate
    if duration < args.min_segment_length:
        # if duration > 0:
        #     shutil.move(srcfn, os.path.join(args.dstdir, 'discard', os.path.basename(srcfn)))
        # else:
        os.remove(srcfn)
        continue
    filelens.append(duration)

    # Optionally split files into fixed length segments
    bn, ext = os.path.basename(srcfn).rsplit('.', 1)
    if args.segment_length != 0:
        dstfn = os.path.join(args.dstdir, bn + '.p%1n.' + ext)
        # ignore SoX stderr as it causes a lot of garbage when file length is shorter than trim length
        cmd = '"{0}" "{1}" "{2}" trim 0 {3} : newfile : restart'
        cmd = cmd.format(SOX, srcfn, dstfn, args.segment_length)
        if args.verbose > 1:
            print(cmd)
        stdout = stderr = subprocess.DEVNULL if args.verbose < 2 else None
        p = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
        procs[srcfn] = p
    else:
        dstfn = os.path.join(args.dstdir, bn + '.p1.' + ext)
        shutil.move(srcfn, dstfn)
    dstfiles.append(dstfn)

# Drop intermediate files
for srcfn in procs.iterpop():
    os.remove(srcfn)

if args.silence_threshold != 0:
    print("Split %d files into a total of %d non-silent segments of lengths: %.3fs min, %.3fs max, %.3fs med" % (len(sourcefiles), len(filelens), min(filelens), max(filelens), np.median(filelens)))

# Grab the list of created files (should be formatted <source-file>.c[0-9]+.p[0-9]+.<ext>)
dstfiles = list(ut.ichain((iglob(srcfn.replace('.p%1n.', '.p*.')) if '.p%1n.' in srcfn else [srcfn]) for srcfn in dstfiles))

# Drop any remaining segments shorter than segment_length
filelens = []
for srcfn in dstfiles:
    with sf.SoundFile(srcfn) as sfh:
        duration = len(sfh) / sfh.samplerate
    if duration < args.min_segment_length:
        # if duration > 0:
        #     shutil.move(srcfn, os.path.join(args.dstdir, 'discard', os.path.basename(srcfn)))
        # else:
        os.remove(srcfn)
        continue
    filelens.append(duration)

#%% Summary
hr, rem = divmod(sum(filelens), 3600)
m, s = divmod(rem, 60)
print("Final output is %d segments (%02d:%02d:%06.3f total)" % (len(filelens), hr, m, s))

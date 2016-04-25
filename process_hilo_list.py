#!/usr/bin/env python

from PyHiLo import *
import sys

if len(sys.argv) != 2:
  raise RuntimeError, "Need an argument: runlist"

print sys.argv[1]
df = pd.read_csv(sys.argv[1], sep=r'\s+', header=None)
df.columns = ['date', 'data', 'laser',  'laser', 'laser', 'laser']
for d in df.date.unique():
    print d, df.data[df.date==d].values[0], df.data[df.date==d].values[1]


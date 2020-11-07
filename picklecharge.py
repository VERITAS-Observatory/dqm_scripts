from PyHiLo import *
import  sys

#Usage:
#python2 picklecharge rootfile.root 0 10 charges.pkl

filename =  str(sys.argv[1])
innerHiGain = sys.argv[2]
samples = sys.argv[3]
picklename = str(sys.argv[4])

hilo = PyHiLo(filename,innerHiGain,samples)
hilo.getAllCharge()
hilo.calcMeanOfMedianHiLo()


pickle.dump(hilo,picklename)


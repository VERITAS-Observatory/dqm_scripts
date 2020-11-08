from PyHiLo import *
import  sys

#Usage:
#python2 picklecharge rootfile.root 0  10  charges.pkl
#Input a stage 2 root file data run used on HiLo Calibration Data
#State 0 or 1 for voltage settings (Usually the first one has an outer hi gain, so 0)
#Samples taken for integration window, for naming purposes
#Name of output file

filename =  str(sys.argv[1])
innerHiGain = sys.argv[2]
samples = sys.argv[3]
picklename = str(sys.argv[4])

hilo = PyHiLo(filename,innerHiGain,samples)
hilo.getAllCharge()
hilo.calcMeanOfMedianHiLo()


with open(picklename,'wb') as picklefile:
    pickle.dump(hilo,picklefile)
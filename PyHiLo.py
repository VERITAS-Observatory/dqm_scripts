#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from optparse import OptionParser
import ROOT
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize
import matplotlib.mlab as mlab
from scipy.stats import norm
import os
import cPickle as pickle
from copy import deepcopy

ROOT.gSystem.Load("$VEGAS/common/lib/libSP24sharedLite.so")

def lin_func(x, a, b):
    return x*a+b

class PyHiLo:
    def __init__(self, filename, innerHiGain, sample=0):
        self.filename = filename
        self.innerHiGain = innerHiGain
        if self.innerHiGain:
            self.MonChanStart=0
            self.MonChanEnd=249
            #self.MonChanEnd=241
            self.testChanStart=250
            self.testChanEnd=498
            #self.testChanEnd=484
        else:
            self.MonChanStart=250
            self.MonChanEnd=498
            #self.MonChanEnd=485
            self.testChanStart=0
            self.testChanEnd=249
            #self.testChanEnd=240
        self.sample=sample
        self.hilo_ratio = np.zeros((4, 499))
        self.profileHi = pd.DataFrame(index=range(4), columns=range(499))
        self.profileLo = pd.DataFrame(index=range(4), columns=range(499))

    def calcMeanOfMedianHiLo(self, numOfMedians=7):
        if not hasattr(self, 'allCharge'):
            print "Reading all charges first..."
            self.getAllCharge()
        numTel = self.allCharge.shape[0]
        numEvt = self.allCharge.shape[2]
        self.numOfMedians = numOfMedians
        self.meanOfMedian=np.zeros(([numTel, numEvt]))
        self.meanLowGainCharge=np.zeros(([numTel, numEvt]))
        for i in range(numTel):
            for j in range(numEvt):
                for k in range(self.numOfMedians):
                    self.meanOfMedian[i][j] += np.median(self.allCharge[i,k*(self.MonChanEnd-self.MonChanStart)/self.numOfMedians+self.MonChanStart:(k+1)*(self.MonChanEnd-self.MonChanStart)/self.numOfMedians+self.MonChanStart, j])
                self.meanOfMedian[i][j] = float(self.meanOfMedian[i][j]/self.numOfMedians)
                #Get mean low gain charge for tel i event j, 
                self.meanLowGainCharge[i][j]=sum(self.allCharge[i,:,j][np.where(self.hiLo[i,:,j]==1)])/sum(self.hiLo[i,:,j])
        return self.meanOfMedian, self.meanLowGainCharge
    
    def getAllCharge(self, outfile=None):
        rootFile = ROOT.VARootIO(self.filename,1)
        calibTree = rootFile.loadTheCalibratedEventTree()
        
        #CalibratedEvents/C/fTelEvents/fChanData/fCharge
        #use "C" tree
        calibEvtData = ROOT.VACalibratedArrayEvent()
        calibTree.SetBranchAddress("C", calibEvtData)
        
        #fArrayEventNum
        #evtNum = []
        totalEvtNum = calibTree.GetEntries()
        self.numberOfEvents = totalEvtNum
        evtNums = np.zeros(totalEvtNum)

        self.allCharge = np.zeros((4, 499, totalEvtNum))
        self.hiLo = np.zeros((4, 499, totalEvtNum))

        evt_count = 0

        for evt in range(totalEvtNum):
            try:
                calibTree.GetEntry(evt)
            except:
                print("Can't get calibrated event number %d" % evt)
                raise
            #evtNum.append(int(calibEvtData.fArrayEventNum))
            try:
                for telID in range(4):
                    fChanData = calibEvtData.fTelEvents.at(telID).fChanData
                    fChanData_iter = ( fChanData.at(i) for i in range(fChanData.size()) )
                    # Save Charge to numpy array
                    for CD in fChanData_iter :
                        chanID = CD.fChanID
                        charge = CD.fCharge
                        self.allCharge[telID][chanID][ent] = CD.fCharge
                        self.hiLo[telID][chanID][ent] = CD.fHiLo
                evtNums[evt_count] = calibEvtData.fArrayEventNum
                evt_count += 1
            except:
                if verbose :
                  print('Something wrong with event: {}'.format(evt))
                  pass

        if outfile !=None:
            pd.DataFrame(self.allCharge).to_csv(outfile, index=False, header=None)
        else:
            return self.allCharge, self.hiLo

    def getFlasherLevels(self):
        self.flasherLevels = np.zeros((4, self.numberOfEvents))
        self.unhandledFlasherLevelsEvents= [[] for i in range(4)]
        #use T4 monitor charge as criteria
        for tel in range(4):
            neg_jumps = np.array(np.where(np.diff(self.meanOfMedian[tel,])<0))[0]
            for i, neg_jump in enumerate(neg_jumps):
                if i==0 and neg_jump>0:
                #figure out flasher level for the first few events
                    for j in range(neg_jump, -1, -1):
                        if j>0 and self.meanOfMedian[tel, j] > self.meanOfMedian[tel, j-1]:
                            self.flasherLevels[tel, j]=j+7-neg_jump
                        elif j==0:
                            self.flasherLevels[tel, j]=j+7-neg_jump
                        else:
                            print "This will never happen."
                    continue
                if i==len(neg_jumps)-1:
                    #dealing with the last cycle
                    #if self.meanOfMedian[tel, neg_jump+1]<=5:
                    for j in range(neg_jump+1, self.numberOfEvents):
                        self.flasherLevels[tel, j]=j-neg_jump-1
                    break
                if neg_jumps[i+1]-neg_jump == 8:
                    #dealing with a regular 7+1 cycle
                    for j in range(neg_jump+1, neg_jumps[i+1]+1):
                        self.flasherLevels[tel, j]=j-neg_jump-1
                else:
                    #dealing with an unusual cycle of < 7+1 levels, 
                    #maybe a pedestal event
                    if self.meanOfMedian[tel, neg_jump+1]<=5 and self.flasherLevels[tel, neg_jump] == 7:
                        #flasher level before jump was 7, 
                        # so should start at 0 and 1 again
                        for j in range(neg_jump+1, neg_jumps[i+1]+1):
                            self.flasherLevels[tel, j]=j-neg_jump-1
                    elif self.meanOfMedian[tel, neg_jump+1]<=5 and self.flasherLevels[tel, neg_jump]<7 and self.meanOfMedian[tel, neg_jump+2] > self.meanOfMedian[tel, neg_jump]:
                        #last cycle finished at flasher level <7, and the first in next cycle has larger charge then the last one, accumulate from the last one
                        self.flasherLevels[tel, neg_jump+1]=0
                        for j in range(neg_jump+2, neg_jumps[i+1]+1):
                            self.flasherLevels[tel, j]=j-neg_jump-1+self.flasherLevels[tel, neg_jump]
                    else:
                        #print "A weird flasher cycle is not handled:"
                        for j in range(neg_jump+1, neg_jumps[i+1]+1):
                            #print "event:", j, "monitor charge:", self.meanOfMedian[tel, j]
                            self.unhandledFlasherLevelsEvents[tel].append(j)
            print "There are "+str(len(self.unhandledFlasherLevelsEvents[tel]))+" events in tel "+str(tel)+" that we cannot determine the flasher levels, see self.unhandledFlasherLevelsEvents."

    def getAllHiLoRatios(self, fitLoRange=[4,5,6,7], fitHiRange=[1,2,3], filebase=None,  fitProfile=True, numberOfProfilesHi=100, numberOfProfilesLo=100, plot=False):
        fitLoRange_init=deepcopy(fitLoRange)
        fitHiRange_init=deepcopy(fitHiRange)
        for tel in [0,1,2,3]:
            for chan in range(self.testChanStart, self.testChanEnd+1):
                if self.hiLo[tel, chan, :].sum() <= 3:
                    print "No low gain events found in test channel", chan, "in tel", tel, "!!!"
                    print "skipping this channel!!!"
                    continue
                # fit ranges are modified by the functions....
                fitLoRange = fitLoRange_init
                fitHiRange = fitHiRange_init
                #print "Initial low gain levels to fit:", fitLoRange
                #print "Initial high gain levels to fit:", fitHiRange
                self.getMonitorVsChannel(telID=tel, chanID=chan, fitLoRange=fitLoRange, fitHiRange=fitHiRange, fitProfile=fitProfile, plot=plot, filebase=filebase, numberOfProfilesHi=numberOfProfilesHi, numberOfProfilesLo=numberOfProfilesLo)
                #self.getMonitorVsChannel(telID=tel, chanID=chan, fitLoRange=fitLoRange, fitHiRange=fitHiRange, filebase=filebase, plot=True)

    def plotFlasherLevelsHist(self, telID):
        fig, ax = plt.subplots(1)
        colors=['r', 'b', 'g', 'm', 'c', 'brown', 'y', 'orange']
        for i in range(8):
            ax.hist(self.meanOfMedian[telID,:][np.where(self.flasherLevels[telID, :]==i)], bins=200, range=[-10,np.max(self.meanOfMedian[telID,:])], color=colors[i], alpha=0.3)
        ax.hist(self.meanOfMedian[telID, self.unhandledFlasherLevelsEvents[telID]], bins=200, range=[-10,np.max(self.meanOfMedian[telID,:])], color='k', alpha=0.5)
        plt.show()

    def getMonitorVsChannel(self, telID=0, chanID=0, plot=False, ax=None, xlim=None, ylim=None, markersize=0.5,
                            fitLoRange=[4, 5, 6, 7], fitHiRange=[1,2,3], filebase=None, fitProfile=True,
                            fmt='eps', numberOfProfilesHi=100, numberOfProfilesLo=100, debug=False):
        if not hasattr(self, 'meanOfMedian'):
            print "You haven't run calcMeanOfMedianHiLo yet..."
            self.calcMeanOfMedianHiLo()
        assert telID>=0 and telID<=3, "Input telID should be 0-3"
        print "Getting monitor vs channel charge for tel",telID, "chan", chanID
        if ax is None and plot:
            fig, ax = plt.subplots(1)
        #for chanID in range(499):
        lowGainEvts=np.where(self.hiLo[telID][chanID][:]==1)
        hiGainEvts=np.where(self.hiLo[telID][chanID][:]==0)
        if fitLoRange is None:
            print "fitLoRange not provided, fitting everything..."
            pars_lo, covs_lo = curve_fit(lin_func, self.meanOfMedian[telID,:][lowGainEvts], self.allCharge[telID][chanID][:][lowGainEvts])
        else:
            #fitLoRange_ = deepcopy(fitLoRange)
            fitLoRange_ = []
            for i, flasher_level_ in enumerate(fitLoRange):
                if sum((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == fitLoRange[i])) >= (4 * sum((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == fitLoRange[i]))):
                    #print "there are",sum((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == fitLoRange_[i])),"low gain events", sum((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == fitLoRange_[i])),"high gain events"
                    #print "less than 80% low gain, skipping"
                    # skip if fewer than 80% of the events are in low gain mode:
                    #del fitLoRange_[i]
                    fitLoRange_.append(flasher_level_)
                else:
                    print "there are",sum((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == fitLoRange[i])),"low gain events", sum((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == fitLoRange[i])),"high gain events"
                    print "less than 80% low gain, skipping flasher level", flasher_level_
            if len(fitLoRange_)<2:
                print fitLoRange_
                print "Fewer than 2 flasher levels are occupied by >80% low gain events, can't fit only one point, quitting..."
                return
            #print "Low gain flasher levels to fit are:", fitLoRange_
            lowGainFitRange=np.where((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] <= fitLoRange_[-1]) & (self.flasherLevels[telID, :] >= fitLoRange_[0]))
            if(len(np.array(lowGainFitRange).flatten())<=2):
                print "No low gain events in tel "+str(telID)+" chan "+str(chanID)+", skipping hilo ratio calculation for this channel."
                return
            if fitProfile:
                #now get a profile from the scatter plot and fit
                profile_list=[]
                for level_j_ in fitLoRange_:
                    # first determine the x_min and x_max for making profiles
                    lowGainFitRange_j=np.where((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == level_j_ ))
                    if level_j_ < fitLoRange_[-1]:
                        lowGainFitRange_jplus1 =np.where((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == (level_j_ + 1) ))
                        charge_lo_j = np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]) - (np.mean(self.meanOfMedian[telID,:][lowGainFitRange_jplus1]) - np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]))/2.
                        charge_hi_j = np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]) + (np.mean(self.meanOfMedian[telID,:][lowGainFitRange_jplus1]) - np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]))/2.
                    elif level_j_ == fitLoRange_[-1]:
                        lowGainFitRange_jminus1 =np.where((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == (level_j_ - 1) ))
                        charge_lo_j = np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]) - (np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]) - np.mean(self.meanOfMedian[telID,:][lowGainFitRange_jminus1]))/2.
                        charge_hi_j = np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]) + (np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]) - np.mean(self.meanOfMedian[telID,:][lowGainFitRange_jminus1]))/2.
                    profile_j = get_profile(self.meanOfMedian[telID,:][lowGainFitRange_j], self.allCharge[telID][chanID][:][lowGainFitRange_j], numberOfProfilesLo, charge_lo_j, charge_hi_j)
                    # get a profile from the j_th flasher level
                    profile_list.append(profile_j)
                # concat all low gain profiles
                self.profileLo.at[telID, chanID] = pd.concat(profile_list)
                self.profileLo.at[telID, chanID] = self.profileLo.at[telID, chanID][self.profileLo.at[telID, chanID]['N']>=20]
                
                if self.profileLo.at[telID, chanID].shape[0]<=2:
                    print "Tel ", telID, " channel ", chanID, " only has 2 or fewer usable low gain profile bins, skipping..."
                    return
                pars_lo, covs_lo = curve_fit(lin_func, self.profileLo.at[telID, chanID]['bincenters'].values, self.profileLo.at[telID, chanID]['ymean'].values, sigma=self.profileLo.at[telID, chanID]['yMeanError'].values)
                #b_lo, db_lo, a_lo, da_lo = ls_lin_fit(self.profileLo['bincenters'].values, self.profileLo['ymean'].values, self.profileLo['yMeanError'].values)
            else:
                pars_lo, covs_lo = curve_fit(lin_func, self.meanOfMedian[telID,:][lowGainFitRange], self.allCharge[telID][chanID][:][lowGainFitRange])
        if fitHiRange is None:
            pars_hi, covs_hi = curve_fit(lin_func, self.meanOfMedian[telID,:][hiGainEvts], self.allCharge[telID][chanID][:][hiGainEvts])
        else:
            #fitHiRange_=deepcopy(fitHiRange)
            fitHiRange_=[]
            for i, flasher_level_ in enumerate(fitHiRange):
                if sum((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == fitHiRange[i])) >= (4 * sum((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == fitHiRange[i]))):
                    # skip if fewer than 80% of the events are in hi gain mode:
                    #del fitHiRange_[i]
                    fitHiRange_.append(flasher_level_)
            if len(fitHiRange_)<2:
                print "Fewer than 2 flasher levels are occupied by >80% high gainevents, can't fit only one point, quitting..."
                return
            #print "High gain flasher levels to fit are:", fitHiRange_
            hiGainFitRange=np.where((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] <= fitHiRange_[-1]) & (self.flasherLevels[telID, :] >= fitHiRange_[0]))
            if(len(np.array(hiGainFitRange).flatten())<=2):
                print "No hi gain events in tel "+str(telID)+" chan "+str(chanID)+", skipping hilo ratio calculation for this channel."
                return
            if fitProfile:
                profile_list=[]
                for level_j_ in fitHiRange_:
                    hiGainFitRange_j=np.where((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == level_j_ ))
                    if level_j_ < fitHiRange_[-1]:
                        hiGainFitRange_jplus1 =np.where((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == (level_j_ + 1) ))
                        charge_lo_j = np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]) - (np.mean(self.meanOfMedian[telID,:][hiGainFitRange_jplus1]) - np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]))/2.
                        charge_hi_j = np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]) + (np.mean(self.meanOfMedian[telID,:][hiGainFitRange_jplus1]) - np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]))/2.
                    elif level_j_ == fitHiRange_[-1]:
                        hiGainFitRange_jminus1 =np.where((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == (level_j_ - 1) ))
                        charge_lo_j = np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]) - (np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]) - np.mean(self.meanOfMedian[telID,:][hiGainFitRange_jminus1]))/2.
                        charge_hi_j = np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]) + (np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]) - np.mean(self.meanOfMedian[telID,:][hiGainFitRange_jminus1]))/2.
                    profile_j = get_profile(self.meanOfMedian[telID,:][hiGainFitRange_j], self.allCharge[telID][chanID][:][hiGainFitRange_j], numberOfProfilesHi, charge_lo_j, charge_hi_j)
                    profile_list.append(profile_j)
                self.profileHi.at[telID, chanID] = pd.concat(profile_list)
                self.profileHi.at[telID, chanID] = self.profileHi.at[telID, chanID][self.profileHi.at[telID, chanID]['N']>=20]
                if self.profileHi.at[telID, chanID].shape[0]<=2:
                    print "Tel ", telID, " channel ", chanID, " only has 2 or fewer usable high gain profile bins, skipping..."
                    return
                pars_hi, covs_hi = curve_fit(lin_func, self.profileHi.at[telID, chanID]['bincenters'].values, self.profileHi.at[telID, chanID]['ymean'].values, sigma=self.profileHi.at[telID, chanID]['yMeanError'].values)
            else:
                pars_hi, covs_hi = curve_fit(lin_func, self.meanOfMedian[telID,:][hiGainFitRange], self.allCharge[telID][chanID][:][hiGainFitRange])
        
        self.hilo_ratio[telID, chanID] = pars_hi[0] / pars_lo[0]
        if plot:
            if fitProfile:
                ax.errorbar(self.profileLo.at[telID, chanID]['bincenters'].values, self.profileLo.at[telID, chanID]['ymean'].values, yerr= self.profileLo.at[telID, chanID]['yMeanError'].values, xerr= self.profileLo.at[telID, chanID]['xerr'].values, color='r', ecolor='r', fmt='none')
                ax.errorbar(self.profileHi.at[telID, chanID]['bincenters'].values, self.profileHi.at[telID, chanID]['ymean'].values, yerr= self.profileHi.at[telID, chanID]['yMeanError'].values, xerr= self.profileHi.at[telID, chanID]['xerr'].values, color='b', ecolor='b', fmt='none')
            else:
                ax.plot(self.meanOfMedian[telID,:][lowGainEvts], self.allCharge[telID][chanID][:][lowGainEvts], 'r.', markersize=markersize)
                ax.plot(self.meanOfMedian[telID,:][hiGainEvts], self.allCharge[telID][chanID][:][hiGainEvts], 'b.', markersize=markersize)
            ax.plot(self.meanOfMedian[telID,:][hiGainEvts], pars_hi[0]*self.meanOfMedian[telID,:][hiGainEvts]+pars_hi[1], 'b-', label="Hi gain slope: "+str("%.2f" % pars_hi[0])+"+/-"+str("%.2f" % np.sqrt(covs_hi[0, 0]))+"\n intercept: "+str("%.2f" % pars_hi[1])+"+/-"+str("%.2f" % np.sqrt(covs_hi[1, 1]))) 
            ax.plot(self.meanOfMedian[telID,:][lowGainEvts], pars_lo[0]*self.meanOfMedian[telID,:][lowGainEvts]+pars_lo[1], 'r-', label="Low gain slope: "+str("%.2f" % pars_lo[0])+"+/-"+str("%.2f" % np.sqrt(covs_lo[0, 0]))+"\n intercept: "+str("%.2f" % pars_lo[1])+"+/-"+str("%.2f" % np.sqrt(covs_lo[1, 1]))+"\n Ratio: "+str("%.2f" % (pars_hi[0]/pars_lo[0]))) 
            ax.set_xlabel("Mean of Median Charge")
            ax.set_ylabel("Channel Charge")
            ax.set_title("T"+str(telID+1)+" chan"+str(chanID))
            if ylim!=None:
                ax.set_ylim(ylim)
            if xlim!=None:
                ax.set_xlim(xlim)
            plt.legend(loc='best', prop={'size':11})
            if filebase is not None:
                plt.savefig(filebase+"tel"+str(telID+1)+"chan"+str(chanID)+'.'+fmt, fmt=fmt)
        #return ax, self.hilo_ratio[telID, chanID]
        if debug:
            print("Now debugging charges for tel %d channel %d..." % (telID, chanID))
            print("First low gain")
            for level_j_ in fitLoRange_:
                print("Low gain flasher level %d" % level_j_)
                lowGainFitRange_j=np.where((self.hiLo[telID][chanID][:]==1) & (self.flasherLevels[telID, :] == level_j_ ))
                fig, ax = plt.subplots(1)
                print("Mean monitor charge: %.2f" % np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j]))
                nMon,binsMon,patchesMon=ax.hist(self.meanOfMedian[telID,:][lowGainFitRange_j],40,normed=1,
                                                facecolor='b',align='mid', label="T"+str(telID+1)+" chan"+str(chanID)+" Monitor")
                stdMon = np.std(self.meanOfMedian[telID,:][lowGainFitRange_j])
                meanMon = np.mean(self.meanOfMedian[telID,:][lowGainFitRange_j])
                fitMonRange = np.where(abs(self.meanOfMedian[telID,:][lowGainFitRange_j]-meanMon)<=stdMon)
                fitMon = self.meanOfMedian[telID,fitMonRange][lowGainFitRange_j]
                (muMon,sigmaMon) = norm.fit(fitMon)
                yMon = norm.pdf(binsMon,loc=muMon,scale=sigmaMon)
                ax.flatten()[telID].plot(binsMon,yMon,'r--',linewidth=2, label="Monitor mean="+str("%.2f" % muMon)+"\nsigma="+str("%.2f" % sigmaMon))
                ax.flatten()[telID].set_ylabel("Normalized counts")
                plt.show()

                print("Mean channel charge: %.2f" % np.mean(self.allCharge[telID][chanID][:][lowGainFitRange_j]))
                n,bins,patches=ax.hist(self.allCharge[telID][chanID][:][lowGainFitRange_j],40,normed=1,
                                       facecolor='g',align='mid', label="T"+str(telID+1)+" chan"+str(chanID)+" Chan Charge")
                std = np.std(self.allCharge[telID][chanID][:][lowGainFitRange_j])
                mean = np.mean(self.allCharge[telID][chanID][:][lowGainFitRange_j])
                fitRange = np.where(abs(self.allCharge[telID][chanID][:][lowGainFitRange_j]-mean)<=std)
                fitC = self.allCharge[telID][chanID][fitRange][lowGainFitRange_j]
                (mu,sigma) = norm.fit(fitC)
                y = norm.pdf(bins,loc=mu,scale=sigma)
                ax.flatten()[telID].plot(bins,y,'r--',linewidth=2, label="Channel mean="+str("%.2f" % mu)+"\nsigma="+str("%.2f" % sigma))
                ax.flatten()[telID].set_ylabel("Normalized counts")
                plt.show()
                raw_input("Press enter to continue...")


    def dumpHiLoRatio(self, filebase='HiLo'):
        for tel in [0,1,2,3]:
            pd.DataFrame(self.hilo_ratio[tel,self.testChanStart:self.testChanEnd+1]).to_csv(filebase+'_T'+str(tel+1)+'.csv', index=False, header=False)

    def plotHiLoRatio(self, filebase=None, fit_norm=False, date=None, runnumber=None):
        fig, ax = plt.subplots(2,2, figsize=(12,9))
        r_ = np.zeros(4)
        dr_ = np.zeros(4)
        for telID in [0,1,2,3]:
            ratios = self.hilo_ratio[telID, self.testChanStart: self.testChanEnd+1]
            ratios = ratios[np.where(ratios>3.)]
            ratios = ratios[np.where(ratios<8.)]
            n,bins,patches=ax.flatten()[telID].hist(ratios,40,normed=fit_norm,facecolor='b',align='mid', label="T"+str(telID+1))
            if fit_norm:
                (mu,sigma) = norm.fit(ratios)
                y = norm.pdf(bins,loc=mu,scale=sigma)
                #ax.hist(ratios, bins=50, normed=1)
                ax.flatten()[telID].plot(bins,y,'r--',linewidth=2, label="mean="+str("%.2f" % mu)+"\nsigma="+str("%.2f" % sigma))
                ax.flatten()[telID].set_ylabel("Normalized counts")
                r_[telID]=mu
                dr_[telID]=sigma
            else:
                x_fit, y_fit, mean_fit, sigma_fit = self.fit_gaussian_hist(bins, n)
                ax.flatten()[telID].plot(x_fit, y_fit,'r--',linewidth=2, label="mean="+str("%.2f" % mean_fit)+"\nsigma="+str("%.2f" % sigma_fit))
                ax.flatten()[telID].set_ylabel("Counts")
                r_[telID]=mean_fit
                dr_[telID]=sigma_fit
            ax.flatten()[telID].legend(loc='best')
            ax.flatten()[telID].set_xlabel("Hi/Lo ratio")
        plt.tight_layout()
        if filebase is None:
            plt.show()
        else:
            plt.savefig(filebase+"HiLoRatioHist.eps", fmt='eps')
            self.dumpHiLoRatio(filebase=filebase+'HiLoRatios')
            self.multipliers_df = pd.DataFrame(np.zeros((1, 10)), 
                  columns=['date', 'run',  
                  'ratio_T1', 'ratio_T2', 'ratio_T3', 'ratio_T4',
                  'dratio_T1', 'dratio_T2', 'dratio_T3', 'dratio_T4'])
            self.multipliers_df.iloc[0] = [date, runnumber, r_[0], r_[1], r_[2], r_[3], dr_[0], dr_[1], dr_[2], dr_[3]]
            self.multipliers_df.to_csv(filebase+'_HiLoMultipliers.csv', index=False, header=False)

            #pd.DataFrame(self.ratios_cut).to_csv(filebase+'HiLoRatios.csv', index=False, header=False)

    def fit_gaussian_hist(self, bins, n):
        """ input is the bin edges and bin content returned by plt.hist. """
        def gaus(x, a, b, c):
            return a * np.exp(-(x - b)**2.0 / (2 * c**2))
        x = [0.5 * (bins[i] + bins[i+1]) for i in xrange(len(bins)-1)]
        y = n
        popt, pcov = optimize.curve_fit(gaus, x, y, p0=(10, np.average(x, weights=n), 0.2))
        print "Fit results", popt
        x_fit = np.linspace(x[0], x[-1], 100)
        y_fit = gaus(x_fit, *popt)
        #returns x, y for plotting, and mean and sigma from fit
        return x_fit, y_fit, popt[1], popt[2]
    
    def dump_pickle(self, filename):
        output = open(filename, 'wb')
        pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close()

### End of class PyHiLo

def load_pickle(filename):
    f_in = file(filename, 'rb')
    hilo = pickle.load(f_in)
    f_in.close()
    return hilo

def processHiLoRun(filename, runnumber, date, number_of_samples, innerHiGain=True, fitProfile=True, numberOfProfilesHi=100, numberOfProfilesLo=100, plot=False, dump=True, read=True, plotTrace=True, overwrite=False):
    filedir = 'hilo'+str(date)
    if read and os.path.exists("hilo"+str(runnumber)+"_"+str(number_of_samples)+"samples.pkl"):
        hilo = load_pickle("hilo"+str(runnumber)+"_"+str(number_of_samples)+"samples.pkl")
        return hilo
    hilo = PyHiLo(filename, innerHiGain, sample=number_of_samples)
    hilo.calcMeanOfMedianHiLo()
    hilo.getFlasherLevels()
    if plotTrace:
        plotAverageTraces(filename, fileout=str(runnumber)+"_"+str(number_of_samples)+"AverageTraces.png")
    if not os.path.isdir(filedir+"/plots_"+str(number_of_samples)+"samples"):
        print "making directory "+filedir+"/plots_"+str(number_of_samples)+"samples"
        os.makedirs(filedir+'/plots_'+str(number_of_samples)+"samples")
    hilo.getAllHiLoRatios(fitLoRange=[4,5,6,7], fitHiRange=[1,2,3], fitProfile=True, numberOfProfilesHi=numberOfProfilesHi, numberOfProfilesLo=numberOfProfilesLo, plot=plot, filebase=filedir+"/plots/hilo"+str(runnumber))
    hilo.plotHiLoRatio(filebase=filedir+'/plots_'+str(runnumber)+"_"+str(number_of_samples)+"samples_"+"unnormed_", date=date, runnumber=runnumber)
    hilo.plotHiLoRatio(filebase=filedir+'/plots_'+str(runnumber)+"_"+str(number_of_samples)+"samples_"+"normed_", fit_norm=True, date=date, runnumber=runnumber)
    if dump and not os.path.exists("hilo"+str(runnumber)+"_"+str(number_of_samples)+"samples.pkl"):
        hilo.dump_pickle("hilo"+str(runnumber)+"_"+str(number_of_samples)+"samples.pkl")
    elif dump and overwrite:
        hilo.dump_pickle("hilo"+str(runnumber)+"_"+str(number_of_samples)+"samples.pkl")
    return hilo

def processBothHiLoRuns(filename1, filename2, runnumber1, runnumber2, date, number_of_samples, innerHiGain1=False, innerHiGain2=True, fitProfile=True, numberOfProfilesHi=100, numberOfProfilesLo=100, plot=False, plotTrace=True, fit_norm=True, xlo=4.5, xhi=7.5):
    print "Processing run "+str(runnumber1)+"..."
    filedir = "hilo"+str(date)
    hilo1 = processHiLoRun(filename1, runnumber1, date, number_of_samples, innerHiGain=innerHiGain1, fitProfile=fitProfile, numberOfProfilesHi=numberOfProfilesHi, numberOfProfilesLo=numberOfProfilesLo, plot=plot, plotTrace=plotTrace, overwrite=True)
    print "Processing run "+str(runnumber2)+"..."
    hilo2 = processHiLoRun(filename2, runnumber2, date, number_of_samples, innerHiGain=innerHiGain2, fitProfile=fitProfile, numberOfProfilesHi=numberOfProfilesHi, numberOfProfilesLo=numberOfProfilesLo, plot=plot, plotTrace=plotTrace, overwrite=True)
    plotBothHilos(hilo1, hilo2, filebase=str(runnumber1)+'_'+str(runnumber2)+'_'+str(number_of_samples)+"samples", fit_norm=fit_norm, xlo=xlo, xhi=xhi)
    getMultipliers(hilo1, filebase="hilo_multipliers/"+str(date)+"_"+str(runnumber1)+"_"+str(number_of_samples)+"sample", fit_norm=fit_norm, date=date, runnumber=runnumber1, sample=number_of_samples)
    getMultipliers(hilo2, filebase="hilo_multipliers/"+str(date)+"_"+str(runnumber2)+"_"+str(number_of_samples)+"sample", fit_norm=fit_norm, date=date, runnumber=runnumber2, sample=number_of_samples)
    return hilo1, hilo2

def getManyMultipliers(hilo_pickles, dates, runnumbers, samples, fit_norm=False):
    multipliers_df = pd.DataFrame(np.zeros((len(dates), 11)),
                                  columns=['date', 'run', 'sample', 
                                  'ratio_T1', 'ratio_T2', 'ratio_T3', 'ratio_T4',
                                  'dratio_T1', 'dratio_T2', 'dratio_T3', 'dratio_T4'])
    i = 0
    for hilo_, date, runnumber, sample in zip(hilo_pickles, dates, runnumbers, samples):
        filebase = "hilo_multipliers/"+str(date)+"_"+str(runnumber)+"_"+str(sample)+"sample"
        hilo = load_pickle(hilo_)
        df_ = getMultipliers(hilo, filebase=filebase, fit_norm=fit_norm, date=date, runnumber=runnumber, sample=sample)
        multipliers_df.iloc[i] = df_.iloc[0]
        i += 1
    return multipliers_df

def getMultipliers(hilo, filebase=None, fit_norm=False, date=None, runnumber=None, sample=7):
    r_ = np.zeros(4)
    dr_ = np.zeros(4)
    for telID in [0,1,2,3]:
        ratios = hilo.hilo_ratio[telID, hilo.testChanStart: hilo.testChanEnd+1]
        ratios = ratios[np.where(ratios>3.)]
        ratios = ratios[np.where(ratios<8.)]
        (mu,sigma) = norm.fit(ratios)
        n,bins,patches=plt.hist(ratios,40,normed=fit_norm,facecolor='b',align='mid', label="T"+str(telID+1))
        if fit_norm:
            y = norm.pdf(bins,loc=mu,scale=sigma)
            r_[telID]=mu
            dr_[telID]=sigma
        else:
            x_fit, y_fit, mean_fit, sigma_fit = fit_gaussian_hist(bins, n)
            r_[telID]=mean_fit
            dr_[telID]=sigma_fit
    if filebase is None:
        return r_, dr_
    else:
        multipliers_df = pd.DataFrame(np.zeros((1, 11)),
                          columns=['date', 'run', 'sample', 
                          'ratio_T1', 'ratio_T2', 'ratio_T3', 'ratio_T4',
                          'dratio_T1', 'dratio_T2', 'dratio_T3', 'dratio_T4'])
        multipliers_df.iloc[0] = [date, runnumber, sample, r_[0], r_[1], r_[2], r_[3], dr_[0], dr_[1], dr_[2], dr_[3]]
        multipliers_df.to_csv(filebase+'_HiLoMultipliers.csv', index=False, header=False)
        hilo.multipliers_df = multipliers_df
        hilo.sample = sample
        hilo.dump_pickle("hilo"+str(runnumber)+"_"+str(sample)+"samples.pkl")
        return multipliers_df


def plotHiloMultipliersFromCSV(filebase="hilo2016-02-01/plots_80480_7samples_unnormed_HiLoRatios_T", fit_norm=False, dump=True, date=None, runnumber=None, sample=None, xlo=4.5, xhi=7.5, filebase2=None):
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    for telID in [0,1,2,3]:
        df_ = pd.read_csv(filebase+str(telID+1)+".csv", header=None)
        ratios = df_.values
        ratios = ratios[np.where(ratios>3.)]
        ratios = ratios[np.where(ratios<8.)]
        
        if filebase2 is not None:
            df2 = pd.read_csv(filebase2+str(telID+1)+".csv", header=None)
            ratios2 = df2.values
            ratios2 = ratios2[np.where(ratios2>3.)]
            ratios2 = ratios2[np.where(ratios2<8.)]
            ratios = np.concatenate((ratios, ratios2))

        (mu,sigma) = norm.fit(ratios)
        n,bins,patches=ax.flatten()[telID].hist(ratios,40,normed=fit_norm,facecolor='b',align='mid', label="T"+str(telID+1))
        if fit_norm:
            y = norm.pdf(bins,loc=mu,scale=sigma)
            #ax.hist(ratios, bins=50, normed=1)
            ax.flatten()[telID].plot(bins,y,'r--',linewidth=2, label="mean="+str("%.2f" % mu)+"\nsigma="+str("%.2f" % sigma))
            ax.flatten()[telID].set_ylabel("Normalized counts")
        else:
            x_fit, y_fit, mean_fit, sigma_fit = fit_gaussian_hist(bins, n)
            sigma_fit = np.abs(sigma_fit)
            ax.flatten()[telID].plot(x_fit, y_fit,'r--',linewidth=2, label="mean="+str("%.2f" % mean_fit)+"\nsigma="+str("%.2f" % sigma_fit))
            ax.flatten()[telID].set_ylabel("Counts")
        ax.flatten()[telID].legend(loc='best')
        ax.flatten()[telID].set_xlabel("Hi/Lo ratio")
        ax.flatten()[telID].set_xlim(xlo, xhi)
    plt.tight_layout()
    if filebase is None:
        plt.show()
    else:
        if filebase2 is not None:
            plt.savefig(filebase[:-2]+"AndNextRunCombined.eps", fmt='eps')
        else:
            plt.savefig(filebase[:-2]+".eps", fmt='eps')

def plotBothHilos(hilo1, hilo2, filebase=None, fit_norm=False, dump=True, date=None, runnumber=None, sample=None, xlo=4.5, xhi=7.5):
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    for telID in [0,1,2,3]:
        ratios1 = hilo1.hilo_ratio[telID, hilo1.testChanStart: hilo1.testChanEnd+1]
        ratios1 = ratios1[np.where(ratios1>3.)]
        ratios1 = ratios1[np.where(ratios1<8.)]
        ratios2 = hilo2.hilo_ratio[telID, hilo2.testChanStart: hilo2.testChanEnd+1]
        ratios2 = ratios2[np.where(ratios2>3.)]
        ratios2 = ratios2[np.where(ratios2<8.)]
        ratios = np.concatenate((ratios1, ratios2))

        (mu,sigma) = norm.fit(ratios)
        n,bins,patches=ax.flatten()[telID].hist(ratios,40,normed=fit_norm,facecolor='b',align='mid', label="T"+str(telID+1))
        if fit_norm:
            y = norm.pdf(bins,loc=mu,scale=sigma)
            #ax.hist(ratios, bins=50, normed=1)
            ax.flatten()[telID].plot(bins,y,'r--',linewidth=2, label="mean="+str("%.2f" % mu)+"\nsigma="+str("%.2f" % sigma))
            ax.flatten()[telID].set_ylabel("Normalized counts")
        else:
            x_fit, y_fit, mean_fit, sigma_fit = fit_gaussian_hist(bins, n)
            ax.flatten()[telID].plot(x_fit, y_fit,'r--',linewidth=2, label="mean="+str("%.2f" % mean_fit)+"\nsigma="+str("%.2f" % sigma_fit))
            ax.flatten()[telID].set_ylabel("Counts")
        ax.flatten()[telID].legend(loc='best')
        ax.flatten()[telID].set_xlabel("Hi/Lo ratio")
        ax.flatten()[telID].set_xlim(xlo, xhi)
    plt.tight_layout()
    if filebase is None:
        plt.show()
    else:
        plt.savefig(filebase+"HiLoRatioHistCombined.eps", fmt='eps')
        #if dump:
        #    multipliers_df = pd.DataFrame(np.zeros((1, 11)),
        #                  columns=['date', 'run', 'sample', 
        #                  'ratio_T1', 'ratio_T2', 'ratio_T3', 'ratio_T4',
        #                  'dratio_T1', 'dratio_T2', 'dratio_T3', 'dratio_T4'])
        #    multipliers_df.iloc[0] = [date, runnumber, sample, r_[0], r_[1], r_[2], r_[3], dr_[0], dr_[1], dr_[2], dr_[3]]
        #    multipliers_df.to_csv(filebase+'_HiLoMultipliers.csv', index=False, header=False)


def fit_gaussian_hist(bins, n):
    """ input is the bin edges and bin content returned by plt.hist. """
    def gaus(x, a, b, c):
        return a * np.exp(-(x - b)**2.0 / (2 * c**2))
    x = [0.5 * (bins[i] + bins[i+1]) for i in xrange(len(bins)-1)]
    y = n
    popt, pcov = optimize.curve_fit(gaus, x, y, p0=(10, np.average(x, weights=n), 0.2))
    print "Fit results", popt
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = gaus(x_fit, *popt)
    #returns x, y for plotting, and mean and sigma from fit
    return x_fit, y_fit, popt[1], popt[2]

def get_profile(x,y,nbins,xmin,xmax):
    df = pd.DataFrame({'x' : x , 'y' : y})
    binedges = xmin + (float(xmax-xmin)/nbins) * np.arange(nbins+1)
    df['bin'] = np.digitize(df['x'],binedges)
    bincenters = xmin + (float(xmax-xmin)/nbins)*np.arange(nbins) + ((xmax-xmin)/(2*nbins))
    ProfileFrame = pd.DataFrame({'bincenters' : bincenters, 'N' : df['bin'].value_counts(sort=False)},index=range(1,nbins+1))
    bins = ProfileFrame.index.values
    for bin in bins:
        ProfileFrame.ix[bin,'ymean'] = df.ix[df['bin']==bin,'y'].mean()
        ProfileFrame.ix[bin,'yStandDev'] = df.ix[df['bin']==bin,'y'].std()
        ProfileFrame.ix[bin,'yMeanError'] = ProfileFrame.ix[bin,'yStandDev'] / np.sqrt(ProfileFrame.ix[bin,'N'])
        ProfileFrame.ix[bin,'xerr']=float(xmax-xmin)/(2*nbins)
    #ax.errorbar(ProfileFrame['bincenters'], ProfileFrame['ymean'], yerr=ProfileFrame['yMeanError'], xerr=ProfileFrame['xerr'], fmt=None) 
    return ProfileFrame

def ls_lin_fit(x, y, yerr):
    # model y = ax + b
    # Least square fit following Hogg, Bovy & Lang (2010) http://arxiv.org/pdf/1008.4686v1.pdf
    # the inverse of yerr*yerr is used to weight y
    # assumes yerr is the correct gaussian uncertainties
    # Code snippet from http://dan.iel.fm/emcee/current/user/line/
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, a_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    return b_ls, np.sqrt(cov[0,0]), a_ls, np.sqrt(cov[1,1])

def plotAverageTraces(infile, fileout=None, log=True):
    rootFile = ROOT.VARootIO(infile,1)
    traceCanvas = ROOT.TCanvas("traceCanvas", "Average Low and High Gain FADC Traces", 0, 0, 1000, 800);
    traceCanvas.Divide(2, 2)
    for i in [0,1,2,3]:
        traceCanvas.cd(i+1);
        AvgTrace_Hi_ = rootFile.loadAnObject("AverageHighGainTrace_Tel"+str(i+1), "Diagnostics/AverageTraces" , True)
        AvgTrace_Lo_ = rootFile.loadAnObject("AverageLowGainTrace_Tel"+str(i+1), "Diagnostics/AverageTraces" , True)
        AvgTrace_Hi_.SetLineColor(4)
        AvgTrace_Hi_.SetTitle("Summed High/Low Gain Traces")
        AvgTrace_Lo_.SetLineColor(2)
        if log:
            ROOT.gPad.SetLogy();
        AvgTrace_Hi_.Draw()
        AvgTrace_Lo_.DrawCopy("same")
    if fileout is not None:
        traceCanvas.SaveAs(fileout);

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("-l","--list",dest="runlist", default=None)
    parser.add_option("--run1",dest="r1", default=None)
    parser.add_option("--run2",dest="r2", default=None)
    parser.add_option("-d","--date",dest="date", default=None)
    parser.add_option("-w","--window",dest="window",default="both")
    #parser.add_option("-inner","--innerHi",dest="innerHi",default=True)
    (options, args) = parser.parse_args()

    if options.runlist is not None:
        df = pd.read_csv(options.runlist, sep=r'\s+', header=None)
        df.columns = ['date', 'data', 'laser',  'laser', 'laser', 'laser']
        for d in df.date.unique():    
            print d, df.data[df.date==d].values[0], df.data[df.date==d].values[1]
            if options.window == "7" or options.window == "both":
                hilo_r1_7, hilo_r2_7 = processBothHiLoRuns(str(df.data[df.date==d].values[0])+"st2_hilo_highWindow7lowWindow7.root", str(df.data[df.date==d].values[1])+"st2_hilo_highWindow7lowWindow7.root", df.data[df.date==d].values[0], df.data[df.date==d].values[1], "hilo"+str(d), 7, innerHiGain1=False, innerHiGain2=True)
            if options.window == "16" or options.window == "both":
                hilo_r1_16, hilo_r2_16 = processBothHiLoRuns(str(df.data[df.date==d].values[0])+"st2_hilo_highWindow16lowWindow16.root", str(df.data[df.date==d].values[1])+"st2_hilo_highWindow16lowWindow16.root", df.data[df.date==d].values[0],df.data[df.date==d].values[1], "hilo"+str(d), 16, innerHiGain1=False, innerHiGain2=True)
    else:
        try:
            if options.window == "7" or options.window == "both":
                hilo_r1_7, hilo_r2_7 = processBothHiLoRuns(str(options.r1)+"st2_hilo_highWindow7lowWindow7.root", str(options.r2)+"st2_hilo_highWindow7lowWindow7.root", options.r1, options.r2, "hilo"+str(options.date), 7, innerHiGain1=False, innerHiGain2=True)
            if options.window == "16" or options.window == "both":
                hilo_r1_16, hilo_r2_16 = processBothHiLoRuns(str(options.r1)+"st2_hilo_highWindow16lowWindow16.root", str(options.r2)+"st2_hilo_highWindow16lowWindow16.root", options.r1, options.r2, "hilo"+str(options.date), 16, innerHiGain1=False, innerHiGain2=True)
        except:
            print "check your options, -l runlist, or -r1 inner hi gain run, -r2 outer hi gain run, -d date"
            raise RuntimeError


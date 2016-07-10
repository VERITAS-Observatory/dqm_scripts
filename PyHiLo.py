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
    
    def getAllCharge(self, outfile=None, maskL2=True, cleaning={'img':5.0,'brd':2.5}, verbose=True):
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

        if maskL2:
            #hard-coded for speed, all numbers are CHANNEL IDs
            neighbor_dict = {0: [1, 2, 3, 4, 5, 6],
                     1: [0, 2, 6, 7, 8, 18],
                     2: [0, 1, 3, 8, 9, 10],
                     3: [0, 2, 4, 10, 11, 12],
                     4: [0, 3, 5, 12, 13, 14],
                     5: [0, 4, 6, 14, 15, 16],
                     6: [0, 1, 5, 16, 17, 18],
                     7: [1, 8, 18, 19, 20, 36],
                     8: [1, 2, 7, 9, 20, 21],
                     9: [2, 8, 10, 21, 22, 23],
                     10: [2, 3, 9, 11, 23, 24],
                     11: [3, 10, 12, 24, 25, 26],
                     12: [3, 4, 11, 13, 26, 27],
                     13: [4, 12, 14, 27, 28, 29],
                     14: [4, 5, 13, 15, 29, 30],
                     15: [5, 14, 16, 30, 31, 32],
                     16: [5, 6, 15, 17, 32, 33],
                     17: [6, 16, 18, 33, 34, 35],
                     18: [1, 6, 7, 17, 35, 36],
                     19: [7, 20, 36, 37, 38, 60],
                     20: [7, 8, 19, 21, 38, 39],
                     21: [8, 9, 20, 22, 39, 40],
                     22: [9, 21, 23, 40, 41, 42],
                     23: [9, 10, 22, 24, 42, 43],
                     24: [10, 11, 23, 25, 43, 44],
                     25: [11, 24, 26, 44, 45, 46],
                     26: [11, 12, 25, 27, 46, 47],
                     27: [12, 13, 26, 28, 47, 48],
                     28: [13, 27, 29, 48, 49, 50],
                     29: [13, 14, 28, 30, 50, 51],
                     30: [14, 15, 29, 31, 51, 52],
                     31: [15, 30, 32, 52, 53, 54],
                     32: [15, 16, 31, 33, 54, 55],
                     33: [16, 17, 32, 34, 55, 56],
                     34: [17, 33, 35, 56, 57, 58],
                     35: [17, 18, 34, 36, 58, 59],
                     36: [7, 18, 19, 35, 59, 60],
                     37: [19, 38, 60, 61, 62, 90],
                     38: [19, 20, 37, 39, 62, 63],
                     39: [20, 21, 38, 40, 63, 64],
                     40: [21, 22, 39, 41, 64, 65],
                     41: [22, 40, 42, 65, 66, 67],
                     42: [22, 23, 41, 43, 67, 68],
                     43: [23, 24, 42, 44, 68, 69],
                     44: [24, 25, 43, 45, 69, 70],
                     45: [25, 44, 46, 70, 71, 72],
                     46: [25, 26, 45, 47, 72, 73],
                     47: [26, 27, 46, 48, 73, 74],
                     48: [27, 28, 47, 49, 74, 75],
                     49: [28, 48, 50, 75, 76, 77],
                     50: [28, 29, 49, 51, 77, 78],
                     51: [29, 30, 50, 52, 78, 79],
                     52: [30, 31, 51, 53, 79, 80],
                     53: [31, 52, 54, 80, 81, 82],
                     54: [31, 32, 53, 55, 82, 83],
                     55: [32, 33, 54, 56, 83, 84],
                     56: [33, 34, 55, 57, 84, 85],
                     57: [34, 56, 58, 85, 86, 87],
                     58: [34, 35, 57, 59, 87, 88],
                     59: [35, 36, 58, 60, 88, 89],
                     60: [19, 36, 37, 59, 89, 90],
                     61: [37, 62, 90, 91, 92, 126],
                     62: [37, 38, 61, 63, 92, 93],
                     63: [38, 39, 62, 64, 93, 94],
                     64: [39, 40, 63, 65, 94, 95],
                     65: [40, 41, 64, 66, 95, 96],
                     66: [41, 65, 67, 96, 97, 98],
                     67: [41, 42, 66, 68, 98, 99],
                     68: [42, 43, 67, 69, 99, 100],
                     69: [43, 44, 68, 70, 100, 101],
                     70: [44, 45, 69, 71, 101, 102],
                     71: [45, 70, 72, 102, 103, 104],
                     72: [45, 46, 71, 73, 104, 105],
                     73: [46, 47, 72, 74, 105, 106],
                     74: [47, 48, 73, 75, 106, 107],
                     75: [48, 49, 74, 76, 107, 108],
                     76: [49, 75, 77, 108, 109, 110],
                     77: [49, 50, 76, 78, 110, 111],
                     78: [50, 51, 77, 79, 111, 112],
                     79: [51, 52, 78, 80, 112, 113],
                     80: [52, 53, 79, 81, 113, 114],
                     81: [53, 80, 82, 114, 115, 116],
                     82: [53, 54, 81, 83, 116, 117],
                     83: [54, 55, 82, 84, 117, 118],
                     84: [55, 56, 83, 85, 118, 119],
                     85: [56, 57, 84, 86, 119, 120],
                     86: [57, 85, 87, 120, 121, 122],
                     87: [57, 58, 86, 88, 122, 123],
                     88: [58, 59, 87, 89, 123, 124],
                     89: [59, 60, 88, 90, 124, 125],
                     90: [37, 60, 61, 89, 125, 126],
                     91: [61, 92, 126, 127, 128, 168],
                     92: [61, 62, 91, 93, 128, 129],
                     93: [62, 63, 92, 94, 129, 130],
                     94: [63, 64, 93, 95, 130, 131],
                     95: [64, 65, 94, 96, 131, 132],
                     96: [65, 66, 95, 97, 132, 133],
                     97: [66, 96, 98, 133, 134, 135],
                     98: [66, 67, 97, 99, 135, 136],
                     99: [67, 68, 98, 100, 136, 137],
                     100: [68, 69, 99, 101, 137, 138],
                     101: [69, 70, 100, 102, 138, 139],
                     102: [70, 71, 101, 103, 139, 140],
                     103: [71, 102, 104, 140, 141, 142],
                     104: [71, 72, 103, 105, 142, 143],
                     105: [72, 73, 104, 106, 143, 144],
                     106: [73, 74, 105, 107, 144, 145],
                     107: [74, 75, 106, 108, 145, 146],
                     108: [75, 76, 107, 109, 146, 147],
                     109: [76, 108, 110, 147, 148, 149],
                     110: [76, 77, 109, 111, 149, 150],
                     111: [77, 78, 110, 112, 150, 151],
                     112: [78, 79, 111, 113, 151, 152],
                     113: [79, 80, 112, 114, 152, 153],
                     114: [80, 81, 113, 115, 153, 154],
                     115: [81, 114, 116, 154, 155, 156],
                     116: [81, 82, 115, 117, 156, 157],
                     117: [82, 83, 116, 118, 157, 158],
                     118: [83, 84, 117, 119, 158, 159],
                     119: [84, 85, 118, 120, 159, 160],
                     120: [85, 86, 119, 121, 160, 161],
                     121: [86, 120, 122, 161, 162, 163],
                     122: [86, 87, 121, 123, 163, 164],
                     123: [87, 88, 122, 124, 164, 165],
                     124: [88, 89, 123, 125, 165, 166],
                     125: [89, 90, 124, 126, 166, 167],
                     126: [61, 90, 91, 125, 167, 168],
                     127: [91, 128, 168, 169, 170, 216],
                     128: [91, 92, 127, 129, 170, 171],
                     129: [92, 93, 128, 130, 171, 172],
                     130: [93, 94, 129, 131, 172, 173],
                     131: [94, 95, 130, 132, 173, 174],
                     132: [95, 96, 131, 133, 174, 175],
                     133: [96, 97, 132, 134, 175, 176],
                     134: [97, 133, 135, 176, 177, 178],
                     135: [97, 98, 134, 136, 178, 179],
                     136: [98, 99, 135, 137, 179, 180],
                     137: [99, 100, 136, 138, 180, 181],
                     138: [100, 101, 137, 139, 181, 182],
                     139: [101, 102, 138, 140, 182, 183],
                     140: [102, 103, 139, 141, 183, 184],
                     141: [103, 140, 142, 184, 185, 186],
                     142: [103, 104, 141, 143, 186, 187],
                     143: [104, 105, 142, 144, 187, 188],
                     144: [105, 106, 143, 145, 188, 189],
                     145: [106, 107, 144, 146, 189, 190],
                     146: [107, 108, 145, 147, 190, 191],
                     147: [108, 109, 146, 148, 191, 192],
                     148: [109, 147, 149, 192, 193, 194],
                     149: [109, 110, 148, 150, 194, 195],
                     150: [110, 111, 149, 151, 195, 196],
                     151: [111, 112, 150, 152, 196, 197],
                     152: [112, 113, 151, 153, 197, 198],
                     153: [113, 114, 152, 154, 198, 199],
                     154: [114, 115, 153, 155, 199, 200],
                     155: [115, 154, 156, 200, 201, 202],
                     156: [115, 116, 155, 157, 202, 203],
                     157: [116, 117, 156, 158, 203, 204],
                     158: [117, 118, 157, 159, 204, 205],
                     159: [118, 119, 158, 160, 205, 206],
                     160: [119, 120, 159, 161, 206, 207],
                     161: [120, 121, 160, 162, 207, 208],
                     162: [121, 161, 163, 208, 209, 210],
                     163: [121, 122, 162, 164, 210, 211],
                     164: [122, 123, 163, 165, 211, 212],
                     165: [123, 124, 164, 166, 212, 213],
                     166: [124, 125, 165, 167, 213, 214],
                     167: [125, 126, 166, 168, 214, 215],
                     168: [91, 126, 127, 167, 215, 216],
                     169: [127, 170, 216, 217, 218, 270],
                     170: [127, 128, 169, 171, 218, 219],
                     171: [128, 129, 170, 172, 219, 220],
                     172: [129, 130, 171, 173, 220, 221],
                     173: [130, 131, 172, 174, 221, 222],
                     174: [131, 132, 173, 175, 222, 223],
                     175: [132, 133, 174, 176, 223, 224],
                     176: [133, 134, 175, 177, 224, 225],
                     177: [134, 176, 178, 225, 226, 227],
                     178: [134, 135, 177, 179, 227, 228],
                     179: [135, 136, 178, 180, 228, 229],
                     180: [136, 137, 179, 181, 229, 230],
                     181: [137, 138, 180, 182, 230, 231],
                     182: [138, 139, 181, 183, 231, 232],
                     183: [139, 140, 182, 184, 232, 233],
                     184: [140, 141, 183, 185, 233, 234],
                     185: [141, 184, 186, 234, 235, 236],
                     186: [141, 142, 185, 187, 236, 237],
                     187: [142, 143, 186, 188, 237, 238],
                     188: [143, 144, 187, 189, 238, 239],
                     189: [144, 145, 188, 190, 239, 240],
                     190: [145, 146, 189, 191, 240, 241],
                     191: [146, 147, 190, 192, 241, 242],
                     192: [147, 148, 191, 193, 242, 243],
                     193: [148, 192, 194, 243, 244, 245],
                     194: [148, 149, 193, 195, 245, 246],
                     195: [149, 150, 194, 196, 246, 247],
                     196: [150, 151, 195, 197, 247, 248],
                     197: [151, 152, 196, 198, 248, 249],
                     198: [152, 153, 197, 199, 249, 250],
                     199: [153, 154, 198, 200, 250, 251],
                     200: [154, 155, 199, 201, 251, 252],
                     201: [155, 200, 202, 252, 253, 254],
                     202: [155, 156, 201, 203, 254, 255],
                     203: [156, 157, 202, 204, 255, 256],
                     204: [157, 158, 203, 205, 256, 257],
                     205: [158, 159, 204, 206, 257, 258],
                     206: [159, 160, 205, 207, 258, 259],
                     207: [160, 161, 206, 208, 259, 260],
                     208: [161, 162, 207, 209, 260, 261],
                     209: [162, 208, 210, 261, 262, 263],
                     210: [162, 163, 209, 211, 263, 264],
                     211: [163, 164, 210, 212, 264, 265],
                     212: [164, 165, 211, 213, 265, 266],
                     213: [165, 166, 212, 214, 266, 267],
                     214: [166, 167, 213, 215, 267, 268],
                     215: [167, 168, 214, 216, 268, 269],
                     216: [127, 168, 169, 215, 269, 270],
                     217: [169, 218, 270, 271, 272, 330],
                     218: [169, 170, 217, 219, 272, 273],
                     219: [170, 171, 218, 220, 273, 274],
                     220: [171, 172, 219, 221, 274, 275],
                     221: [172, 173, 220, 222, 275, 276],
                     222: [173, 174, 221, 223, 276, 277],
                     223: [174, 175, 222, 224, 277, 278],
                     224: [175, 176, 223, 225, 278, 279],
                     225: [176, 177, 224, 226, 279, 280],
                     226: [177, 225, 227, 280, 281, 282],
                     227: [177, 178, 226, 228, 282, 283],
                     228: [178, 179, 227, 229, 283, 284],
                     229: [179, 180, 228, 230, 284, 285],
                     230: [180, 181, 229, 231, 285, 286],
                     231: [181, 182, 230, 232, 286, 287],
                     232: [182, 183, 231, 233, 287, 288],
                     233: [183, 184, 232, 234, 288, 289],
                     234: [184, 185, 233, 235, 289, 290],
                     235: [185, 234, 236, 290, 291, 292],
                     236: [185, 186, 235, 237, 292, 293],
                     237: [186, 187, 236, 238, 293, 294],
                     238: [187, 188, 237, 239, 294, 295],
                     239: [188, 189, 238, 240, 295, 296],
                     240: [189, 190, 239, 241, 296, 297],
                     241: [190, 191, 240, 242, 297, 298],
                     242: [191, 192, 241, 243, 298, 299],
                     243: [192, 193, 242, 244, 299, 300],
                     244: [193, 243, 245, 300, 301, 302],
                     245: [193, 194, 244, 246, 302, 303],
                     246: [194, 195, 245, 247, 303, 304],
                     247: [195, 196, 246, 248, 304, 305],
                     248: [196, 197, 247, 249, 305, 306],
                     249: [197, 198, 248, 250, 306, 307],
                     250: [198, 199, 249, 251, 307, 308],
                     251: [199, 200, 250, 252, 308, 309],
                     252: [200, 201, 251, 253, 309, 310],
                     253: [201, 252, 254, 310, 311, 312],
                     254: [201, 202, 253, 255, 312, 313],
                     255: [202, 203, 254, 256, 313, 314],
                     256: [203, 204, 255, 257, 314, 315],
                     257: [204, 205, 256, 258, 315, 316],
                     258: [205, 206, 257, 259, 316, 317],
                     259: [206, 207, 258, 260, 317, 318],
                     260: [207, 208, 259, 261, 318, 319],
                     261: [208, 209, 260, 262, 319, 320],
                     262: [209, 261, 263, 320, 321, 322],
                     263: [209, 210, 262, 264, 322, 323],
                     264: [210, 211, 263, 265, 323, 324],
                     265: [211, 212, 264, 266, 324, 325],
                     266: [212, 213, 265, 267, 325, 326],
                     267: [213, 214, 266, 268, 326, 327],
                     268: [214, 215, 267, 269, 327, 328],
                     269: [215, 216, 268, 270, 328, 329],
                     270: [169, 216, 217, 269, 329, 330],
                     271: [217, 272, 330, 331, 332, 396],
                     272: [217, 218, 271, 273, 332, 333],
                     273: [218, 219, 272, 274, 333, 334],
                     274: [219, 220, 273, 275, 334, 335],
                     275: [220, 221, 274, 276, 335, 336],
                     276: [221, 222, 275, 277, 336, 337],
                     277: [222, 223, 276, 278, 337, 338],
                     278: [223, 224, 277, 279, 338, 339],
                     279: [224, 225, 278, 280, 339, 340],
                     280: [225, 226, 279, 281, 340, 341],
                     281: [226, 280, 282, 341, 342, 343],
                     282: [226, 227, 281, 283, 343, 344],
                     283: [227, 228, 282, 284, 344, 345],
                     284: [228, 229, 283, 285, 345, 346],
                     285: [229, 230, 284, 286, 346, 347],
                     286: [230, 231, 285, 287, 347, 348],
                     287: [231, 232, 286, 288, 348, 349],
                     288: [232, 233, 287, 289, 349, 350],
                     289: [233, 234, 288, 290, 350, 351],
                     290: [234, 235, 289, 291, 351, 352],
                     291: [235, 290, 292, 352, 353, 354],
                     292: [235, 236, 291, 293, 354, 355],
                     293: [236, 237, 292, 294, 355, 356],
                     294: [237, 238, 293, 295, 356, 357],
                     295: [238, 239, 294, 296, 357, 358],
                     296: [239, 240, 295, 297, 358, 359],
                     297: [240, 241, 296, 298, 359, 360],
                     298: [241, 242, 297, 299, 360, 361],
                     299: [242, 243, 298, 300, 361, 362],
                     300: [243, 244, 299, 301, 362, 363],
                     301: [244, 300, 302, 363, 364, 365],
                     302: [244, 245, 301, 303, 365, 366],
                     303: [245, 246, 302, 304, 366, 367],
                     304: [246, 247, 303, 305, 367, 368],
                     305: [247, 248, 304, 306, 368, 369],
                     306: [248, 249, 305, 307, 369, 370],
                     307: [249, 250, 306, 308, 370, 371],
                     308: [250, 251, 307, 309, 371, 372],
                     309: [251, 252, 308, 310, 372, 373],
                     310: [252, 253, 309, 311, 373, 374],
                     311: [253, 310, 312, 374, 375, 376],
                     312: [253, 254, 311, 313, 376, 377],
                     313: [254, 255, 312, 314, 377, 378],
                     314: [255, 256, 313, 315, 378, 379],
                     315: [256, 257, 314, 316, 379, 380],
                     316: [257, 258, 315, 317, 380, 381],
                     317: [258, 259, 316, 318, 381, 382],
                     318: [259, 260, 317, 319, 382, 383],
                     319: [260, 261, 318, 320, 383, 384],
                     320: [261, 262, 319, 321, 384, 385],
                     321: [262, 320, 322, 385, 386, 387],
                     322: [262, 263, 321, 323, 387, 388],
                     323: [263, 264, 322, 324, 388, 389],
                     324: [264, 265, 323, 325, 389, 390],
                     325: [265, 266, 324, 326, 390, 391],
                     326: [266, 267, 325, 327, 391, 392],
                     327: [267, 268, 326, 328, 392, 393],
                     328: [268, 269, 327, 329, 393, 394],
                     329: [269, 270, 328, 330, 394, 395],
                     330: [217, 270, 271, 329, 395, 396],
                     331: [271, 332, 396, 397, 462],
                     332: [271, 272, 331, 333, 397, 398],
                     333: [272, 273, 332, 334, 398, 399],
                     334: [273, 274, 333, 335, 399, 400],
                     335: [274, 275, 334, 336, 400, 401],
                     336: [275, 276, 335, 337, 401, 402],
                     337: [276, 277, 336, 338, 402, 403],
                     338: [277, 278, 337, 339, 403, 404],
                     339: [278, 279, 338, 340, 404, 405],
                     340: [279, 280, 339, 341, 405, 406],
                     341: [280, 281, 340, 342, 406, 407],
                     342: [281, 341, 343, 407, 408],
                     343: [281, 282, 342, 344, 408, 409],
                     344: [282, 283, 343, 345, 409, 410],
                     345: [283, 284, 344, 346, 410, 411],
                     346: [284, 285, 345, 347, 411, 412],
                     347: [285, 286, 346, 348, 412, 413],
                     348: [286, 287, 347, 349, 413, 414],
                     349: [287, 288, 348, 350, 414, 415],
                     350: [288, 289, 349, 351, 415, 416],
                     351: [289, 290, 350, 352, 416, 417],
                     352: [290, 291, 351, 353, 417, 418],
                     353: [291, 352, 354, 418, 419],
                     354: [291, 292, 353, 355, 419, 420],
                     355: [292, 293, 354, 356, 420, 421],
                     356: [293, 294, 355, 357, 421, 422],
                     357: [294, 295, 356, 358, 422, 423],
                     358: [295, 296, 357, 359, 423, 424],
                     359: [296, 297, 358, 360, 424, 425],
                     360: [297, 298, 359, 361, 425, 426],
                     361: [298, 299, 360, 362, 426, 427],
                     362: [299, 300, 361, 363, 427, 428],
                     363: [300, 301, 362, 364, 428, 429],
                     364: [301, 363, 365, 429, 430],
                     365: [301, 302, 364, 366, 430, 431],
                     366: [302, 303, 365, 367, 431, 432],
                     367: [303, 304, 366, 368, 432, 433],
                     368: [304, 305, 367, 369, 433, 434],
                     369: [305, 306, 368, 370, 434, 435],
                     370: [306, 307, 369, 371, 435, 436],
                     371: [307, 308, 370, 372, 436, 437],
                     372: [308, 309, 371, 373, 437, 438],
                     373: [309, 310, 372, 374, 438, 439],
                     374: [310, 311, 373, 375, 439, 440],
                     375: [311, 374, 376, 440, 441],
                     376: [311, 312, 375, 377, 441, 442],
                     377: [312, 313, 376, 378, 442, 443],
                     378: [313, 314, 377, 379, 443, 444],
                     379: [314, 315, 378, 380, 444, 445],
                     380: [315, 316, 379, 381, 445, 446],
                     381: [316, 317, 380, 382, 446, 447],
                     382: [317, 318, 381, 383, 447, 448],
                     383: [318, 319, 382, 384, 448, 449],
                     384: [319, 320, 383, 385, 449, 450],
                     385: [320, 321, 384, 386, 450, 451],
                     386: [321, 385, 387, 451, 452],
                     387: [321, 322, 386, 388, 452, 453],
                     388: [322, 323, 387, 389, 453, 454],
                     389: [323, 324, 388, 390, 454, 455],
                     390: [324, 325, 389, 391, 455, 456],
                     391: [325, 326, 390, 392, 456, 457],
                     392: [326, 327, 391, 393, 457, 458],
                     393: [327, 328, 392, 394, 458, 459],
                     394: [328, 329, 393, 395, 459, 460],
                     395: [329, 330, 394, 396, 460, 461],
                     396: [271, 330, 331, 395, 461, 462],
                     397: [331, 332, 398],
                     398: [332, 333, 397, 399],
                     399: [333, 334, 398, 400, 463],
                     400: [334, 335, 399, 401, 463, 464],
                     401: [335, 336, 400, 402, 464, 465],
                     402: [336, 337, 401, 403, 465, 466],
                     403: [337, 338, 402, 404, 466, 467],
                     404: [338, 339, 403, 405, 467, 468],
                     405: [339, 340, 404, 406, 468],
                     406: [340, 341, 405, 407],
                     407: [341, 342, 406],
                     408: [342, 343, 409],
                     409: [343, 344, 408, 410],
                     410: [344, 345, 409, 411, 469],
                     411: [345, 346, 410, 412, 469, 470],
                     412: [346, 347, 411, 413, 470, 471],
                     413: [347, 348, 412, 414, 471, 472],
                     414: [348, 349, 413, 415, 472, 473],
                     415: [349, 350, 414, 416, 473, 474],
                     416: [350, 351, 415, 417, 474],
                     417: [351, 352, 416, 418],
                     418: [352, 353, 417],
                     419: [353, 354, 420],
                     420: [354, 355, 419, 421],
                     421: [355, 356, 420, 422, 475],
                     422: [356, 357, 421, 423, 475, 476],
                     423: [357, 358, 422, 424, 476, 477],
                     424: [358, 359, 423, 425, 477, 478],
                     425: [359, 360, 424, 426, 478, 479],
                     426: [360, 361, 425, 427, 479, 480],
                     427: [361, 362, 426, 428, 480],
                     428: [362, 363, 427, 429],
                     429: [363, 364, 428],
                     430: [364, 365, 431],
                     431: [365, 366, 430, 432],
                     432: [366, 367, 431, 433, 481],
                     433: [367, 368, 432, 434, 481, 482],
                     434: [368, 369, 433, 435, 482, 483],
                     435: [369, 370, 434, 436, 483, 484],
                     436: [370, 371, 435, 437, 484, 485],
                     437: [371, 372, 436, 438, 485, 486],
                     438: [372, 373, 437, 439, 486],
                     439: [373, 374, 438, 440],
                     440: [374, 375, 439],
                     441: [375, 376, 442],
                     442: [376, 377, 441, 443],
                     443: [377, 378, 442, 444, 487],
                     444: [378, 379, 443, 445, 487, 488],
                     445: [379, 380, 444, 446, 488, 489],
                     446: [380, 381, 445, 447, 489, 490],
                     447: [381, 382, 446, 448, 490, 491],
                     448: [382, 383, 447, 449, 491, 492],
                     449: [383, 384, 448, 450, 492],
                     450: [384, 385, 449, 451],
                     451: [385, 386, 450],
                     452: [386, 387, 453],
                     453: [387, 388, 452, 454],
                     454: [388, 389, 453, 455, 493],
                     455: [389, 390, 454, 456, 493, 494],
                     456: [390, 391, 455, 457, 494, 495],
                     457: [391, 392, 456, 458, 495, 496],
                     458: [392, 393, 457, 459, 496, 497],
                     459: [393, 394, 458, 460, 497, 498],
                     460: [394, 395, 459, 461, 498],
                     461: [395, 396, 460, 462],
                     462: [331, 396, 461],
                     463: [399, 400, 464],
                     464: [400, 401, 463, 465],
                     465: [401, 402, 464, 466],
                     466: [402, 403, 465, 467],
                     467: [403, 404, 466, 468],
                     468: [404, 405, 467],
                     469: [410, 411, 470],
                     470: [411, 412, 469, 471],
                     471: [412, 413, 470, 472],
                     472: [413, 414, 471, 473],
                     473: [414, 415, 472, 474],
                     474: [415, 416, 473],
                     475: [421, 422, 476],
                     476: [422, 423, 475, 477],
                     477: [423, 424, 476, 478],
                     478: [424, 425, 477, 479],
                     479: [425, 426, 478, 480],
                     480: [426, 427, 479],
                     481: [432, 433, 482],
                     482: [433, 434, 481, 483],
                     483: [434, 435, 482, 484],
                     484: [435, 436, 483, 485],
                     485: [436, 437, 484, 486],
                     486: [437, 438, 485],
                     487: [443, 444, 488],
                     488: [444, 445, 487, 489],
                     489: [445, 446, 488, 490],
                     490: [446, 447, 489, 491],
                     491: [447, 448, 490, 492],
                     492: [448, 449, 491],
                     493: [454, 455, 494],
                     494: [455, 456, 493, 495],
                     495: [456, 457, 494, 496],
                     496: [457, 458, 495, 497],
                     497: [458, 459, 496, 498],
                     498: [459, 460, 497]}

        evt_count = 0

        for evt in range(totalEvtNum):
            try:
                calibTree.GetEntry(evt)
            except:
                print("Can't get calibrated event number %d" % evt)
                raise
            #evtNum.append(int(calibEvtData.fArrayEventNum))
            try:
                snrStorage = np.zeros(500)
                for telID in range(4):
                    snrStorage.fill(0.0)
                    brd_candidate_index = np.array([],dtype=int)
                    fChanData = calibEvtData.fTelEvents.at(telID).fChanData
                    fChanData_iter = ( fChanData.at(i) for i in range(fChanData.size()) )
                    # Save Charge to numpy array
                    for CD in fChanData_iter :
                        chanID = CD.fChanID
                        charge = CD.fCharge
                        SNR    = CD.fSignalToNoise
                        self.allCharge[telID][chanID][ent] = CD.fCharge
                        self.hiLo[telID][chanID][ent] = CD.fHiLo
                        snrStorage[chanID] = SNR
                        if cleaning is not None:
                            if SNR < cleaning['brd']:
                                allCharge[telID][chanID][evt_count] = 0
                            elif SNR < cleaning['img']:
                                brd_candidate_index = np.append(brd_candidate_index,chanID)
                        if cleaning is not None:
                            print("Cleaning the images...")
                            for chanID in brd_candidate_index:
                                passed  = False
                                for neighbor in neighbor_dict[chanID]:
                                    if snrStorage[neighbor] > cleaning['img']:
                                        passed = True
                                        break
                                if not passed:
                                    allCharge[telID][chanID][evt_count] = 0
                        # Average over neighboring pixels for L2-masked pixels
                        if maskL2:
                            for l2chan in l2channels[telID]:
                                if (l2chan != 499):
                                    allCharge[telID][l2chan][evt_count] = np.mean(allCharge[telID,neighbor_dict[l2chan],evt_count])
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
                            fmt='eps', numberOfProfilesHi=100, numberOfProfilesLo=100, debug=False, save_debug=None):
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
                mon_j = self.meanOfMedian[telID][lowGainFitRange_j]
                stdMon = np.std(mon_j)
                meanMon = np.mean(mon_j)
                nMon,binsMon,patchesMon=ax.hist(mon_j,40,normed=1,
                                                facecolor='b',align='mid', label="T"+str(telID+1)+" chan"+str(chanID)+" Monitor \nmean="+str("%.2f" % meanMon)+"\nsigma="+str("%.2f" % stdMon))
                fitMonRange = np.where(abs(mon_j-meanMon)<=stdMon)
                fitMon = mon_j[fitMonRange]
                (muMon,sigmaMon) = norm.fit(fitMon)
                yMon = norm.pdf(binsMon,loc=muMon,scale=sigmaMon)
                ax.plot(binsMon,yMon,'r--',linewidth=2, label="Fit Monitor mean="+str("%.2f" % muMon)+"\nsigma="+str("%.2f" % sigmaMon))
                ax.set_ylabel("Normalized counts")
                plt.legend(loc='best')
                if save_debug is not None:
                    plt.savefig("lowGainT"+str(telID+1)+"Chan"+str(chanID)+"FlasherLevel"+str(level_j_)+"_"+"MonCharge.png")
                plt.show()

                print("Mean channel charge: %.2f" % np.mean(self.allCharge[telID][chanID][lowGainFitRange_j]))
                fig, ax = plt.subplots(1)
                allC_j = self.allCharge[telID][chanID][lowGainFitRange_j]
                std = np.std(allC_j)
                mean = np.mean(allC_j)
                fitRange = np.where(abs(allC_j-mean)<=std)
                fitC = allC_j[fitRange]
                n,bins,patches=ax.hist(allC_j,40,normed=1,
                                       facecolor='g',align='mid', label="T"+str(telID+1)+" chan"+str(chanID)+" Chan Charge \nmean="+str("%.2f" % mean)+"\nsigma="+str("%.2f" %std))
                (mu,sigma) = norm.fit(fitC)
                y = norm.pdf(bins,loc=mu,scale=sigma)
                ax.plot(bins,y,'r--',linewidth=2, label="Channel mean="+str("%.2f" % mu)+"\nsigma="+str("%.2f" % sigma))
                ax.set_ylabel("Normalized counts")
                plt.legend(loc='best')
                if save_debug is not None:
                    plt.savefig("lowGainT"+str(telID+1)+"Chan"+str(chanID)+"FlasherLevel"+str(level_j_)+"_"+"ChanCharge.png")
                plt.show()

            print("Then high gain")
            for level_j_ in fitHiRange_:
                print("High gain flasher level %d" % level_j_)
                hiGainFitRange_j=np.where((self.hiLo[telID][chanID][:]==0) & (self.flasherLevels[telID, :] == level_j_ ))
                fig, ax = plt.subplots(1)
                print("Mean monitor charge: %.2f" % np.mean(self.meanOfMedian[telID,:][hiGainFitRange_j]))
                mon_j = self.meanOfMedian[telID][hiGainFitRange_j]
                stdMon = np.std(mon_j)
                meanMon = np.mean(mon_j)
                nMon,binsMon,patchesMon=ax.hist(mon_j,40,normed=1,
                                                facecolor='b',align='mid', label="T"+str(telID+1)+" chan"+str(chanID)+" Monitor \nmean="+str("%.2f" % meanMon)+"\nsigma="+str("%.2f" % stdMon))
                fitMonRange = np.where(abs(mon_j-meanMon)<=stdMon)
                fitMon = mon_j[fitMonRange]
                (muMon,sigmaMon) = norm.fit(fitMon)
                yMon = norm.pdf(binsMon,loc=muMon,scale=sigmaMon)
                ax.plot(binsMon,yMon,'r--',linewidth=2, label="Fit Monitor mean="+str("%.2f" % muMon)+"\nsigma="+str("%.2f" % sigmaMon))
                ax.set_ylabel("Normalized counts")
                plt.legend(loc='best')
                if save_debug is not None:
                    plt.savefig("highGainT"+str(telID+1)+"Chan"+str(chanID)+"FlasherLevel"+str(level_j_)+"_"+"MonCharge.png")
                plt.show()

                print("Mean channel charge: %.2f" % np.mean(self.allCharge[telID][chanID][hiGainFitRange_j]))
                fig, ax = plt.subplots(1)
                allC_j = self.allCharge[telID][chanID][hiGainFitRange_j]
                std = np.std(allC_j)
                mean = np.mean(allC_j)
                fitRange = np.where(abs(allC_j-mean)<=std)
                fitC = allC_j[fitRange]
                n,bins,patches=ax.hist(allC_j,40,normed=1,
                                       facecolor='g',align='mid', label="T"+str(telID+1)+" chan"+str(chanID)+" Chan Charge \nmean="+str("%.2f" % mean)+"\nsigma="+str("%.2f" %std))
                (mu,sigma) = norm.fit(fitC)
                y = norm.pdf(bins,loc=mu,scale=sigma)
                ax.plot(bins,y,'r--',linewidth=2, label="Channel mean="+str("%.2f" % mu)+"\nsigma="+str("%.2f" % sigma))
                ax.set_ylabel("Normalized counts")
                plt.legend(loc='best')
                if save_debug is not None:
                    plt.savefig("highGainT"+str(telID+1)+"Chan"+str(chanID)+"FlasherLevel"+str(level_j_)+"_"+"ChanCharge.png")
                plt.show()

            #raw_input("Press enter to continue...")


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


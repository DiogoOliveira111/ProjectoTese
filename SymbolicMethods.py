import numpy as np
from AuxiliaryMethods import detect_peaks
import math

# Connotation Methods
def DiffC(s, t, signs=['N', 'F', 'P']):
    # Quantization of the derivative.
    # TODO: Implement a better way of selecting chars
    ds1 = np.diff(s)
    x = np.empty(len(s), dtype=str)
    thr = (np.max(ds1) - np.min(ds1)) * t
    x[np.where(ds1 <= -thr)[0]] = signs[0]
    x[np.where(np.all([ds1 <= thr, ds1 >= -thr], axis=0))[0]] = signs[1]
    x[np.where(thr <= ds1)[0]] = signs[2]
    x[-1] = x[-2]

    return x


def Diff2C(s, t, symbols=['N', 'F', 'P']):
    # Quantization of the derivative.
    # TODO: Implement a better threshold methodology.
    dds1 = np.diff(np.diff(s))
    x = np.empty(len(s), dtype=str)
    thr = (np.max(dds1) - np.min(dds1)) * t
    x[np.where(dds1 <= -thr)[0]] = symbols[0]
    x[np.where(np.all([dds1 <= thr, dds1 >= -thr], axis=0))[0]] = symbols[1]
    x[np.where(thr <= dds1)[0]] = symbols[2]
    x[-1] = x[-2]

    return x


def RiseAmp(Signal, t):
    # detect all valleys
    val = detect_peaks(Signal, valley=True)
    # final pks array
    pks = []
    thr = ((np.max(Signal) - np.min(Signal)) * t) + np.mean(Signal)
    # array of amplitude of rising with size of signal
    risingH = np.zeros(len(Signal))
    Rise = np.array([])

    for i in range(0, len(val) - 1):
        # piece of signal between two successive valleys
        wind = Signal[val[i]:val[i + 1]]
        # find peak between two minimums
        pk = detect_peaks(wind, mph=0.1 * max(Signal))
        # print(pk)
        # if peak is found:
        if (len(pk) > 0):
            # append peak position
            pks.append(val[i] + pk)
            # calculate rising amplitude
            # val=np.array(val)
            # pk=np.array(pk)
            # risingH=np.array(risingH)
            wind=np.array(wind)
            risingH[val[i]:val[i + 1]] = [wind[pk] - Signal[val[i]] for a in range(val[i], val[i + 1])]
            Rise = np.append(Rise, (wind[pk] - Signal[val[i]]) > thr)

    risingH = np.array(risingH > thr).astype(int)
    Rise = Rise.astype(int)

    return risingH

def AmpC(s, t, p='>'):

    if( s=='None'):
        print('something missing here')
    thr = (float(np.max(s) - np.min(s)) * t) + np.min(s)
    if (p == '<'):
        s1 = (s <= (thr)) * 1
    elif (p == '>'):
        s1 = (s >= (thr)) * 1

    return s1

def absAmp(s, t1, t2):
    s1=[]
    print(t1)
    print(t2)
    for i in s:
        if i>=t1 and i<=t2:
            s1.append("1")
        else:
            s1.append("0")
    return s1

def findDuplicates(s):
    seen=[]
    final_String=[]
    for i in s:
        if i not in seen:
            seen.append(i)
            final_String.append("0")
        else:
            final_String.append("1")
    return final_String

def isFlat(s):
    s1= []

    for i in range(len(s)):
        if i==len(s)-1:
            if s[1]==s[i-1]:
                s1.append("1")
            else:
                s1.append("0")
        else:
            if s[i]==s[i+1]:
                s1.append("1")
            else:
                s1.append("0")
    return s1

def guideFixed(s):
    maxS=max(s)
    thr=1/maxS
    return thr

def maximum(s):
    max_value=max(s)
    max_index=[i for i, j in enumerate(s) if j == max_value]
    s1=np.zeros(len(s), dtype=int)
    print(s1)
    for i in max_index:
        s1[i]="1"
    print('giverer')
    print(s1)
    return s1


def minimum(s):
    min_value  = min(i for i in s if i > 0) #para garantir que o minimo nao é o zero
    min_index = [i for i, j in enumerate(s) if j == min_value]
    s1 = np.zeros(len(s), dtype=int)
    for i in min_index:
        s1[i] = "1"
    return s1

def average(s, thr):
    s1 = np.zeros(len(s), dtype=int)
    stringArray = np.array(s)
    mean=stringArray[stringArray.nonzero()].mean() #para nao contar os zeros na media

    thrS = thr * mean + mean  # threshold superior
    thrI = thr * mean - mean  # threshold inferior
    for i in range(len(s)):
        if s[i]==0:
            s1[i]="0"
        elif s[i] <= thrS and s[i] >= thrI:
            s1[i]="1"

    return s1

def averageUnique(s, thr):
    s1=np.zeros(len(s), dtype=int)
    uniqueValues=set(s)
    uniqueValues.remove(0) #retirar os zeros que nao entram na media
    print(uniqueValues)
    stringList=list(uniqueValues)
    mean=sum(stringList)/len(stringList)
    thrS=thr*mean+mean #threshold superior
    thrI=thr*mean-mean #threshold inferior
    for i in range(len(s)):
        if s[i]==0:
            s1[i]="0"
        elif s[i]<=thrS and s[i]>=thrI:
            s1[i]="1"

    return s1

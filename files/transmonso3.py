import argparse
import numpy as np
import os
import pandas as pd
from gwtrigfind import find_trigger_files
from gwpy.table import EventTable
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
import datetime
from gwpy.time import tconvert
from gwpy.time import from_gps
from gwpy.time import to_gps
from gwpy.segments import Segment
from datetime import timedelta
from itertools import islice
from scipy.signal import correlate
import matplotlib.pyplot as plt
from gwpy.table import Table
import time
import h5py
from gwpy.segments import SegmentList
from gwpy.timeseries import TimeSeriesDict
from gwdetchar import scattering
from gwpy.segments import DataQualityFlag



parser = argparse.ArgumentParser(description='Vetoes potential scattering triggers')
parser.add_argument('gpsstart', type=int,
                    help='gpstime of the start of the day.')
parser.add_argument('gpsend', type=int,
                    help='gpstime of the end of the day.')
parser.add_argument('-L','--flow',type=int,
		     help='lower limit of the bandpass frequency for transmons')
parser.add_argument('-H','--fhigh',type=int,
		     help='upper limit of the bandpass frequency for tranmons')
parser.add_argument('-S','--sigma',type=int,
		     help='standard deviation above mean for tranmon blrms')
#parser.add_argument('-s','--snr',type=int,
#		     help='minimum snr threshold on h(t) triggers to be vetoed by transmons')
parser.add_argument('-o', '--output-dir', type=os.path.abspath,
                    default=os.curdir,
                    help='output directory for analysis, default: %(default)s')
args = parser.parse_args()

starttime = args.gpsstart
endtime = args.gpsend
thres = args.sigma
#snrt = args.snr
filename = 'Xtransmonveto'+str(starttime)+'-'+str(endtime)+'.csv'

print(starttime)
print(endtime)
print(args.flow)
print(args.fhigh)
print(thres)
#print(snrt)

# first collect all the observing segments
seglist = DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1',starttime,endtime).active
## Padding the segments.
newsegs = [Segment(i[0]+20,i[1]-20) for i in seglist]


## Observing duration
obs = []
for i in newsegs:
   obs.append(i[1] - i[0])
totalobs = np.sum(obs)

print(newsegs)
print(totalobs)


cachegds = find_trigger_files('L1:GDS-CALIB_STRAIN','omicron',starttime,endtime,ext = "h5")
t = EventTable.read(cachegds,format = 'hdf5', path = 'triggers', columns = ['time','frequency','snr']).filter('snr>15','snr<200','frequency<60','frequency>10')

print("h(t) triggers read")

tintrans=[]
flaga=[]

for i in newsegs:
   if i[1] - i[0] > 4:
       XTR = TimeSeries.fetch('L1:ASC-X_TR_B_NSUM_OUT_DQ',i[0],i[1])
       XTR = XTR.whiten(4,2).bandpass(args.flow,args.fhigh).rms(1)
       print("TimeSeries fetched and whitened for the segment {0}".format(i))
       highxtr = XTR > np.mean(XTR) + thres*np.std(XTR)
       flag = highxtr.to_dqflag(round=True)
       flag = flag.coalesce()
       for i in flag.active:
           flaga.append(i)
       else:
           pass

#print(flaga)

print("Transmon segs constructed")

## Finding coincidence of transmon blrms segments with the darm triggers.
for i in flaga:
   for j in t['time']:
      if j in i:
         tintrans.append(j)
         break

#print(tintrans)

## Efficiency
eff = round((len(tintrans)*100)/len(t),2)

## Deadtime
dur = []
for i in flaga:
   dur.append(i[1]-i[0])
totaldur = np.sum(dur)

deadtime = round(float((totaldur)*100/totalobs),2)

print(eff)
print(deadtime)

# Efficiency over deadtime
eod = round(eff/deadtime,2)

print("The efficiency over deadtime is {0}".format(eod))


dfgspy = pd.read_csv('gspy-o3a_scat_tf_img_ordmodel.csv')
dfgspy.drop_duplicates('GPStime',inplace=True)
dfgspy = dfgspy.sort_values(by='GPStime',inplace=True)
dfgspy = dfgspy.reset_index(drop=True)
## Finding how many vetoed triggers match with gravityspy.

dfgspy1 = dfgspy[(dfgspy['time']<endtime) & (dfgspy['time']>starttime)].reset_index(drop=True)

print("The number of Scattering triggers caught by GravitySpy between {0} and {1} between snr 15 and 200 and with confidence above 0.9 are {2}".format(starttime,endtime,len(dfgspy1)))

k=0
for i in dfgspy1[i'GPStime']:
   for j in tintrans:
      if abs(i-j)<0.5:
         k+=1
         break

print("The X end tranmon caught {0} triggers from {1} to {2} and {3} of them"
" match with the triggers identified as scattering by gravityspy for the same duration".format(len(tintrans),starttime,endtime,k))




dft = t.to_pandas()
freqintrans=[]
snrintrans=[]
for i in tintrans:
   freqintrans.append(dft[dft['time']==i]['frequency'].values[0])
   snrintrans.append(dft[dft['time']==i]['snr'].values[0])


with open('transmoninfo.csv','a+') as f:
   f.write('For the gpstime from {0} to {1}, the total h(t) triggers above snr 15 and peak frequency between 10 and 60 are {2}, total number of transmon segments are {3},'
' efficiency and deadtime of transmon segments are {4} and {5}. The X end transmon caught {6} triggers and {7} of these match triggers identified as scattering by gravityspy. Gravityspy'
' found a total of {8} triggers during the same duration with snr above 15 and no filter on peakfrequency.'.format(from_gps(starttime),from_gps(endtime),len(t),len(flaga),eff,deadtime,len(tintrans),k,len(dfgspy1))+'\n')

dfintrans = pd.DataFrame(list(zip(tintrans,snrintrans,freqintrans)),columns=['time','snr','frequency'])
print(dfintrans)

dfintrans.to_csv(filename,index=False)

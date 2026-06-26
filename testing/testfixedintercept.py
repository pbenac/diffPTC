import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import yaml
import scipy

from gain import *

with open('datasets/260623_IMG_lightleak.yaml') as f:
        config = yaml.safe_load(f)

head = config['dataset']['head']
date = config['dataset']['date']
datapath = config['dataset']['datapath']
outpath = config['dataset']['outpath']
setname = config['dataset']['name']


X0 = config['roi']['x_start']
X1 = config['roi']['x_end']
Y0 = config['roi']['y_start']
Y1 = config ['roi']['y_end']

modes = ['slow52', 'fast10', 'fast06']
for mode in modes:
    name = config[mode]['name']
    itime = config[mode]['frametime']
    n1 = config[mode]['illuminated']['n1']
    n2 = config[mode]['illuminated']['n2']
    detrend = config[mode]['detrend']
   

intercept = 1.0
dict = gain_diff_forceIntercept(head, date, n1, n2, datapath, X0, X1, Y0, Y1, intercept, detrend=False, fixed_xmax=1500, _quiet=False)
print(dict)
plt.figure()
plt.plot(dict['S'], dict['V'], marker='o')
xs = np.linspace(np.min(dict['S']), np.max(dict['S']))
plt.plot(xs, xs*dict['slope']+dict['inter'])
plt.xlabel('Signal')
plt.ylabel('Variance')
plt.title('Fast0.6, gain'+str(dict['gain'])+" and intercept "+str(dict['inter']))
plt.savefig('testfixedintercept.png')

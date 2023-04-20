# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:45:59 2014

@author: andrew
"""

import os
import numpy as np
import pylab as pl
from scipy.io import wavfile
from scipy.signal import resample

fig = pl.figure(num=None, figsize=(3.34646*3/2, 3.34646*1/3), facecolor='w', 
                frameon=True)

Fs, s = wavfile.read('block1_trial002.wav')
ax = fig.add_subplot(1, 3, 1, aspect='auto', frameon=True)
ax.set_xticks(np.arange(0, 6, 1))
ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=7)
ax.set_yticks(np.arange(0, 5001, 1000))
ax.set_yticklabels([0, 1, 2, 3, 4, 5], fontsize=7)
ax.specgram(s[:,1], NFFT=2048, Fs=Fs, noverlap=2048*7/8., cmap='gray', 
            clim=[0, 0.000001])
ax.set_ylim(0, 5000)
ax.set_xlim(0, 5)

Fs, s = wavfile.read('block2_trial001.wav')
ax = fig.add_subplot(1, 3, 2, aspect='auto', frameon=True)
ax.set_xticks(np.arange(0, 6, 1))
ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=7)
ax.set_yticks(np.arange(0, 5001, 1000))
ax.set_yticklabels([0, 1, 2, 3, 4, 5], fontsize=7)
ax.specgram(s[:,1], NFFT=2048, Fs=Fs, noverlap=2048*7/8., cmap='gray', 
            clim=[0, 0.000001])
ax.set_ylim(0, 5000)
ax.set_xlim(0, 5)

Fs, s = wavfile.read('block3_trial001.wav')
ax = fig.add_subplot(1, 3, 3, aspect='auto', frameon=True)
ax.set_xticks(np.arange(0, 6, 1))
ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=7)
ax.set_yticks(np.arange(0, 5001, 1000))
ax.set_yticklabels([0, 1, 2, 3, 4, 5], fontsize=7)
ax.specgram(s[:,1], NFFT=2048, Fs=Fs, noverlap=2048*7/8., cmap='gray', 
            clim=[0, 0.000001])
ax.set_ylim(0, 5000)
ax.set_xlim(0, 5)
            
figname = 'example_stimuli'
fig.savefig(figname + '.svg', format='svg', transparent=True)
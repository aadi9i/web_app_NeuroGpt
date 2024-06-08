# -*- coding: utf-8 -*-
"""
Muse LSL Example Auxiliary Tools

These functions perform the lower-level operations involved in buffering,
epoching, and transforming EEG data into frequency bands

@author: Cassani
"""

import os
import sys
from tempfile import gettempdir
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from scipy.signal import butter, lfilter, lfilter_zi


NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')


def epoch(data, samples_epoch, samples_overlap=0):
    """Extract epochs from a time series.

    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [wlength_samples, n_channels, n_epochs]

    Args:
        data (numpy.ndarray or list of lists): data [n_samples, n_channels]
        samples_epoch (int): window length in samples
        samples_overlap (int): Overlap between windows in samples

    Returns:
        (numpy.ndarray): epoched data of shape
    """

    if isinstance(data, list):
        data = np.array(data)

    n_samples, n_channels = data.shape

    samples_shift = samples_epoch - samples_overlap

    n_epochs = int(
        np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs


def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG.

    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-6
    ind_theta1, = np.where((f >= 4) & (f <= 6))
    meanTheta1= np.mean(PSD[ind_theta1, :], axis=0)
    # Theta 6.5-8
    ind_theta2, = np.where((f >= 6.5) & (f <= 8))
    meanTheta2 = np.mean(PSD[ind_theta2, :], axis=0)
    # Alpha 8.5-10
    ind_alpha1, = np.where((f >= 8.5) & (f <= 10))
    meanAlpha1 = np.mean(PSD[ind_alpha1, :], axis=0)
    # Alpha 10.5-13
    ind_alpha2, = np.where((f >= 10.5) & (f <= 13))
    meanAlpha2 = np.mean(PSD[ind_alpha2, :], axis=0)
    # Beta 13.5-18
    ind_beta1, = np.where((f >= 13.5) & (f < 18))
    meanBeta1 = np.mean(PSD[ind_beta1, :], axis=0)
    # Beta 18.5-30
    ind_beta2, = np.where((f >= 18.5) & (f < 30))
    meanBeta2 = np.mean(PSD[ind_beta2, :], axis=0)
    # Gamma 30.5-40
    ind_gam1 = np.where((f>=30.5)&(f<=40))
    meanGamma1 =  np.mean(PSD[ind_gam1, :], axis=0).flatten()
     # Gamma 40-49.5
    ind_gam2 = np.where((f>=40)&(f<=49.5))
    meanGamma2 =  np.mean(PSD[ind_gam2, :], axis=0).flatten()

    feature_vector = np.concatenate((meanGamma1,meanGamma2, meanTheta1,meanTheta2, meanAlpha1,meanAlpha2,
                                     meanBeta1,meanBeta2), axis=0)#eanDelta

    feature_vector = np.log10(feature_vector)

    return feature_vector


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs):
    """
    Call compute_feature_vector for each EEG epoch
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_band_powers(epochs[:, :, i_epoch], fs).T
            # Initialize feature_matrix
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))

        feature_matrix[i_epoch, :] = compute_band_powers(
            epochs[:, :, i_epoch], fs).T

    return feature_matrix


def get_feature_names(ch_names):
    """Generate the name of the features.

    Args:
        ch_names (list): electrode names

    Returns:
        (list): feature names
    """
    bands = ['gamma1','gamma2', 'theta1','theta2', 'alpha1','theta2', 'beta1','beta2']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim != data_buffer.shape[1]:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)
    print("data_buffer shape:", data_buffer.shape)
    print("new_data shape:", new_data.shape)
    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer
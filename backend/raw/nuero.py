# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

# .2
import numpy as np  # Module that simplifies computations on matrices
import time
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import muselsl
import csv, os
# Handy little enum to make code more readable


class Band:
    Gamma1 = 0
    Gamma2 = 1
    Theta1 = 2
    Theta2 = 3
    Alpha1 = 4
    Alpha2 = 5
    Beta1 = 6
    Beta2 = 7


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]
duration = 30  # Seconds

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """
    muselsl.stream(None,backend="bluemuse")
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 8))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        file = 'my_data.csv'
        indices = [' Theta1: 0', ' Theta1: 1', ' Theta1: 2', ' Theta1: 3', 
                   ' Theta2: 0', ' Theta2: 1', ' Theta2: 2', ' Theta2: 3', 
                   ' Alpha1: 0', ' Alpha1: 1', ' Alpha1: 2', ' Alpha1: 3',
                    ' Alpha2: 0', ' Alpha2: 1', ' Alpha2: 2', ' Alpha2: 3',
                    ' Beta1: 0', ' Beta1: 1', ' Beta1: 2', ' Beta1: 3',
                    ' Beta2: 0', ' Beta2: 1', ' Beta2: 2', ' Beta2: 3',
                    'Gamma1: 0', 'Gamma1: 1', 'Gamma1: 2', 'Gamma1: 3',
                    'Gamma2: 0', 'Gamma2: 1', 'Gamma2: 2', 'Gamma2: 3']

        with open(file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(indices)

            # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
            start_time = time.time()
            while time.time() - start_time < duration:
            # while True:
                list_to_append = []
                """ 3.1 ACQUIRE DATA """
                # Obtain EEG data from the LSL stream
                eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
                print("EEG DATA SHAPE", np.array(eeg_data).shape)
                for INDEX_CHANNEL in [[0],[1],[2],[3]]:
                    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

                    # Update EEG buffer with the new data
                    eeg_buffer, filter_state = utils.update_buffer(
                        eeg_buffer, ch_data, notch=True,
                        filter_state=filter_state)

                    """ 3.2 COMPUTE BAND POWERS """
                    # Get newest samples from the buffer
                    data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
                    # Compute band powers
                    band_powers = utils.compute_band_powers(data_epoch, fs)
                    band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
                    # Compute the average band powers for all epochs in buffer
                    # This helps to smooth out noise
                    smooth_band_powers = np.mean(band_buffer, axis=0)

                    print(f"For channel number {INDEX_CHANNEL[0]} :")
                    print('Gamma1: ', band_powers[Band.Gamma1],'Gamma2: ', band_powers[Band.Gamma2], ' Theta1: ', band_powers[Band.Theta1],' Theta2: ', band_powers[Band.Theta2],
                        ' Alpha1: ', band_powers[Band.Alpha1],' Alpha2: ', band_powers[Band.Alpha2], ' Beta1: ', band_powers[Band.Beta1],' Beta2: ', band_powers[Band.Beta2])
                    list_to_append.extend([band_powers[Band.Theta1], band_powers[Band.Theta2], band_powers[Band.Alpha1], band_powers[Band.Alpha2], band_powers[Band.Beta1], band_powers[Band.Beta2], band_powers[Band.Gamma1], band_powers[Band.Gamma2]])
                writer.writerow(list_to_append)

    except KeyboardInterrupt:
        print('Closing!')
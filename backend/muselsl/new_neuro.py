import numpy as np  # Module that simplifies computations on matrices
import time
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import muselsl
import csv

def extract_data_to_csv(duration = 30):
    Gamma1 = 0
    Gamma2 = 1
    Theta1 = 2
    Theta2 = 3
    Alpha1 = 4
    Alpha2 = 5
    Beta1 = 6
    Beta2 = 7

    BUFFER_LENGTH = 5
    EPOCH_LENGTH = 1
    OVERLAP_LENGTH = 0.8
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
    duration = 10

    muselsl.stream(None,backend="bluemuse")
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0: raise RuntimeError('Can\'t find EEG stream.')
    inlet = StreamInlet(streams[0], max_chunklen=12)

    info = inlet.info()
    fs = int(info.nominal_srate())
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 8))

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
        start_time = time.time()
        while time.time() - start_time < duration:
            list_to_append = []
            eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            # print("EEG DATA SHAPE", np.array(eeg_data).shape)
            for INDEX_CHANNEL in [[0],[1],[2],[3]]:
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
                eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)
                data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
                band_powers = utils.compute_band_powers(data_epoch, fs)
                band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))

                # print(f"For channel number {INDEX_CHANNEL[0]} :")
                # print('Gamma1: ', band_powers[Gamma1],'Gamma2: ', band_powers[Gamma2], ' Theta1: ', band_powers[Theta1],' Theta2: ', band_powers[Theta2],
                #     ' Alpha1: ', band_powers[Alpha1],' Alpha2: ', band_powers[Alpha2], ' Beta1: ', band_powers[Beta1],' Beta2: ', band_powers[Beta2])
                list_to_append.extend([band_powers[Theta1], band_powers[Theta2], band_powers[Alpha1], band_powers[Alpha2], band_powers[Beta1], band_powers[Beta2], band_powers[Gamma1], band_powers[Gamma2]])
            writer.writerow(list_to_append)


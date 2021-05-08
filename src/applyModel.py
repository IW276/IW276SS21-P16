import os

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm
import pickle

import torch

"""
###### Extract a log-mel spectrogram for each audio file in the dataset and store it into a Pandas DataFrame along with its class and fold label.
"""
HOP_LENGTH = 512        # number of samples between successive frames
WINDOW_LENGTH = 512     # length of the window in samples
N_MEL = 128             # number of Mel bands to generate
SOUND_DURATION = 2.95   # fixed duration of an audio excerpt in seconds


def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        # compute a mel-scaled spectrogram
        melspectrogram = librosa.feature.melspectrogram(y=audio,
                                                        sr=sampling_rate,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH,
                                                        n_mels=N_MEL)

        # convert a power spectrogram to decibel units (log-mel spectrogram)
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

        melspectrogram_length = melspectrogram_db.shape[1]

        # pad or fix the length of spectrogram
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db,
                                                        size=num_of_samples,
                                                        axis=1,
                                                        constant_values=(0, -80.0))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None

    return melspectrogram_db

def apply(net, data_entry):
    x = torch.tensor(data_entry).to(self.device)

    outputs = net(x)

    predictions = torch.argmax(outputs, dim=1)

    return predictions

file_path = "../data/fold1/7061-6-0-0.wav"
audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')
melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)
# convert into a Pandas DataFrame
input_data = pd.DataFrame(melspectrogram)
print(input_data.shape)

if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")
print(device)

model_name = "mediocreModel.pt"
net = torch.load(model_name, map_location={'cuda:0': 'cpu'})
net = net.to(device)
result = apply(net, input_data)
print(result)

import os

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
###### Extract a log-mel spectrogram for each audio file in the dataset and store it into a Pandas DataFrame along with its class and fold label.
"""
HOP_LENGTH = 512        # number of samples between successive frames
WINDOW_LENGTH = 512     # length of the window in samples
N_MEL = 128             # number of Mel bands to generate
SOUND_DURATION = 2.95   # fixed duration of an audio excerpt in seconds

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)

        self.device = device

    def forward(self, x):
        # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.fc2(x)

        return x

    def fit(self, train_loader, epochs, val_loader=None):
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

        for epoch in range(epochs):
            self.train()

            print("\nEpoch {}/{}".format(epoch+1, epochs))

            with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
                for step, batch in enumerate(train_loader):
                    X_batch = batch['spectrogram'].to(self.device)
                    y_batch = batch['label'].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        # forward + backward
                        outputs = self.forward(X_batch)
                        batch_loss = self.criterion(outputs, y_batch)
                        batch_loss.backward()

                        # update the parameters
                        self.optimizer.step()

                    pbar.update(1)

            # model evaluation - train data
            train_loss, train_acc = self.evaluate(train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

        return history

    def predict(self, X):
        self.eval()

        with torch.no_grad():
            outputs = self.forward(X)

        return outputs

    def evaluate(self, data_loader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(data_loader.batch_size).to(self.device)

        for step, batch in enumerate(data_loader):
            X_batch = batch['spectrogram'].to(self.device)
            y_batch = batch['label'].to(self.device)

            outputs = self.predict(X_batch)

            # get batch loss
            loss = self.criterion(outputs, y_batch)
            running_loss = running_loss + loss

            # calculate batch accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = (predictions == y_batch).float().sum()
            running_acc = running_acc + torch.div(correct_predictions, batch_size)

        loss = running_loss.item() / (step+1)
        accuracy = running_acc.item() / (step+1)

        return loss, accuracy

def normalize_data(train_df, test_df):
    # compute the mean and std (pixel-wise)
    mean = train_df['melspectrogram'].mean()
    std = np.std(np.stack(train_df['melspectrogram']), axis=0)

    # normalize train set
    train_spectrograms = (np.stack(train_df['melspectrogram']) - mean) / std
    train_labels = train_df['label'].to_numpy()
    train_folds = train_df['fold'].to_numpy()
    train_df = pd.DataFrame(zip(train_spectrograms, train_labels, train_folds), columns=['melspectrogram', 'label', 'fold'])

    # normalize test set
    test_spectrograms = (np.stack(test_df['melspectrogram']) - mean) / std
    test_labels = test_df['label'].to_numpy()
    test_folds = test_df['fold'].to_numpy()
    test_df = pd.DataFrame(zip(test_spectrograms, test_labels, test_folds), columns=['melspectrogram', 'label', 'fold'])

    return train_df, test_df

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

def normalize(spectrogram, mean, std):
    return (np.stack(spectrogram) - mean) / std

def apply(net, data_entry, device):
    x = data_entry.to(device)

    outputs = net(x)

    predictions = torch.argmax(outputs, dim=1)

    return predictions


input_dir = "./testdata/"

if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")
print("using device: ", device)

ix_to_class = { 0:"Klimaanlage",
                1:"Autohupe",
                2:"Kinder Spielen",
                3:"Hund bellt",
                4:"Bohren",
                5:"Motorgeräusche",
                6:"Schuss",
                7:"Presslufthammer",
                8:"Sirene",
                9:"Straßenmusik"}


norm_mean = np.load("mean.npy")
norm_std = np.load("std.npy")

net = Net(device)
model_name = "mediocreModel_0.pt"
net.load_state_dict(torch.load(model_name, map_location={'cuda:0': 'cpu'}))
net.eval() # set dropout and batch normalization to evaluation mode
net = net.to(device)

output_file_name = "./out/ergebnisse.txt"
with open(output_file_name, 'w', encoding='utf-8') as output_file:
    for file in os.listdir(input_dir):
        file_path = input_dir + file
        noise_length = librosa.get_duration(filename=file_path)

        classifications = []
        output_file.write("Klassifiziere " + file_path + "\n")
        for i in range(int(noise_length / SOUND_DURATION)+1):
            print("Klassifiziere [%d]: offset:%.2f" % (i, SOUND_DURATION*i))

            audio, sample_rate = librosa.load(file_path, offset=SOUND_DURATION*i, duration=SOUND_DURATION, res_type='kaiser_fast')
            melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)

            melspectrogram = normalize(melspectrogram, norm_mean, norm_std)

            # (n_samples, channels, height, width) -> one sample on one channel
            input_data = torch.tensor(melspectrogram).unsqueeze(0).unsqueeze(0)

            result = apply(net, input_data, device)

            # print(file_path + " > '" + ix_to_class[result.item()] + "'")
            classifications.append(result.item())
            output_file.write("  [Offset: %.2f] > %s\n" % (SOUND_DURATION*i, ix_to_class[result.item()]))
        output_file.write("  Häufigste Klasse: " + str(ix_to_class[max(classifications, key=classifications.count)]) + "\n")
        output_file.write("\n")

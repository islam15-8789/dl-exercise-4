import torch
import torchvision as tv
from data import ChallengeDataset

# from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
# import model
import pandas as pd
from sklearn.model_selection import train_test_split



# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

# create an instance of our ResNet model
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO

dataframe = pd.read_csv("data.csv", sep=';')
challenge_dataset = ChallengeDataset(dataframe, "train")
batch_size = 100
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(challenge_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    challenge_dataset, batch_size=batch_size, sampler=train_sampler
)
validation_loader = torch.utils.data.DataLoader(
    challenge_dataset, batch_size=batch_size, sampler=valid_sampler
)

def __show_data(data_loader, num_of_data=6):
    for images, image_names in data_loader:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(tv.utils.make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break
    # plt.show()
    plt.savefig('losses.png')


__show_data(validation_loader)

# go, go, go... call fit on trainer
# res = #TODO

# # plot the results
# plt.plot(np.arange(len(res[0])), res[0], label='train loss')
# plt.plot(np.arange(len(res[1])), res[1], label='val loss')
# plt.yscale('log')
# plt.legend()
# plt.savefig('losses.png')
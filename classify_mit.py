from cgi import test
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
# You might not have tqdm, which gives you nice progress bars
from tqdm.notebook import tqdm
import os
import copy
import json


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")


input_size = 224
batch_size = 8
shuffle_datasets = True
data_dir = "static/"
is_labelled = False 
generate_labels = True
k = 3

def get_dataloaders():
    # transform the image when loading them.
    # can change transforms on the training set.
    
    # For now, we resize/crop the image to the correct input size for our network,
    # then convert it to a [C,H,W] tensor, then normalize it to values with a given mean/stdev. These normalization constants
    # are derived from aggregating lots of data and happen to produce better results.
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    } 
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}
    # Create training and validation dataloaders
    # Never shuffle the test set
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else True, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict 

dataloaders = get_dataloaders() 

def label_number_to_name(lbl_ix):
    return dataloaders['val'].dataset.classes[lbl_ix]


def dataset_labels_to_names(dataset_labels, dataset_name):
    # dataset_name is one of 'train','test','val'
    dataset_root = os.path.join(data_dir, dataset_name)
    found_files = []
    for parentdir, subdirs, subfns in os.walk(dataset_root):
        parentdir_nice = os.path.relpath(parentdir, dataset_root)
        found_files.extend([os.path.join(parentdir_nice, fn) for fn in subfns if fn.endswith('.jpg') or fn.endswith('.jpeg')])
    # Sort alphabetically, this is the order that our dataset will be in
    found_files.sort()
    # Now we have two parallel arrays, one with names, and the other with predictions
    assert len(found_files) == len(dataset_labels), "Found more files than we have labels"
    preds = {os.path.basename(found_files[i]):list(map(label_number_to_name, dataset_labels[i])) for i in range(len(found_files))}
    return preds

def get_prediction():
    model = models.resnet18(pretrained=False) # UPDATE WITH BEST MODEL
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 25) # num_classes = 25
    model.load_state_dict(torch.load("Resnet_Non_Augmented_Weights_Best.pt", map_location=torch.device('cpu')))

    predicted_labels = []

    model.eval()

    # print(dataloaders["test"])

    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # with torch.set_grad_enabled(False):
        #     outputs = model(inputs)
        #     _, preds = torch.topk(outputs, k=3, dim=1)
        #     nparr = preds.cpu().detach().numpy()
        #     predicted_labels.extend([list(nparr[i]) for i in range(len(nparr))])

    # test_labels_js = dataset_labels_to_names(predicted_labels, "test")

    # output_test_labels = "test_set_predictions"
    # output_salt_number = 0

    # output_label_dir = "."

    # while os.path.exists(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number))):
    #     output_salt_number += 1
    #     # Find a filename that doesn't exist
    

    # with open(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number)), "w") as f:
    #     json.dump(test_labels_js, f, sort_keys=True, indent=4)

# print("Wrote predictions to:\n%s" % os.path.abspath(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number))))
get_prediction()
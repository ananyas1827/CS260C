from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd

# create a figure with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(16, 4))
DNNs = ['MLP', 'GNN']
types = ['Classification', 'Regression']
split_type = '-1'
for row, type in enumerate(types):
  dnn = DNNs[0]
  # get csv files in type
  csv_files = os.listdir(f'{dnn}/{type}/')
  # filter non-csv files
  csv_files = [file for file in csv_files if file.endswith('.csv')]
  # remove csv files with split_type in name
  csv_files = [file for file in csv_files if split_type in file] 
  for col, file in enumerate(csv_files):
    print(row, col)
    lr = file.split('_')[3]
    # rm .csv
    lr = float(lr[:-4])
    df = pd.read_csv(f'{dnn}/{type}/{file}') 
    epochs = df['epoch'].values
    if type == 'Classification':
      train_acc = df['train_accuracy'].values
      test_acc = df['test_accuracy'].values
      # red for train, blue for test
      axes[row, col].plot(epochs, train_acc, color='red', label='Train Accuracy')
      axes[row, col].plot(epochs, test_acc, color='blue', label='Test Accuracy')
      # use scientific notation for lr
      axes[row, col].set_title(f'{dnn} {type} lr={lr:0.0e}')
      axes[row, col].legend()
    else:
      train_loss = df['train_loss'].values
      test_loss = df['test_loss'].values
      # red for train, blue for test
      axes[row, col].plot(epochs, train_loss, color='red', label='Train Loss')
      axes[row, col].plot(epochs, test_loss, color='blue', label='Test Loss')
      axes[row, col].set_title(f'{dnn} {type} lr={lr:0.0e}')
      axes[row, col].legend()
  dnn = DNNs[1]
  # get csv files in type
  csv_files = os.listdir(f'{dnn}/{type}/')
  # filter non-csv files
  csv_files = [file for file in csv_files if file.endswith('.csv')]
  csv_files = [file for file in csv_files if split_type in file] 

  for col, file in enumerate(csv_files, start=1):
    print(row, col)
    df = pd.read_csv(f'{dnn}/{type}/{file}') 
    lr = float(file.split('_')[3])
    n_nodes = file.split('_')[4]
    # rm .csv
    n_nodes = int(n_nodes[:-4])
    epochs = df['epoch'].values
    if type == 'Classification':
      train_acc = df['train_accuracy'].values
      test_acc = df['test_accuracy'].values
      # red for train, blue for test
      axes[row, col].plot(epochs, train_acc, color='red', label='Train Accuracy')
      axes[row, col].plot(epochs, test_acc, color='blue', label='Test Accuracy')
      axes[row, col].set_title(f'{dnn}-{n_nodes} {type} lr={lr:0.0e}')
      axes[row, col].legend()
    else:
      train_loss = df['train_loss'].values
      test_loss = df['test_loss'].values
      # red for train, blue for test
      axes[row, col].plot(epochs, train_loss, color='red', label='Train Loss')
      axes[row, col].plot(epochs, test_loss, color='blue', label='Test Loss')
      axes[row, col].set_title(f'{dnn}-{n_nodes} {type} lr={lr:0.0e}')
      axes[row, col].legend()
plt.tight_layout()
# save pdf
plt.savefig(f'results_split_{split_type}.pdf')
# save png
plt.savefig(f'results_split_{split_type}.png')

        
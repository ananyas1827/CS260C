import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Dataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import torch
import copy
import os
import argparse

# get ../current working directory
prj_path = os.path.abspath(os.path.join(os.getcwd(), "../"))

# set random seed 
torch.manual_seed(0)

# Define a custom dataset class
class MyDataset(Dataset):
	def __init__(self, data_list):
		super(MyDataset, self).__init__()
		self.data = data_list

	def len(self):
		return len(self.data)

	def get(self, idx):
		return self.data[idx]

def prepare_datasets(shapes, train_percent, average_n_features, batch_size=64):
	# new geometric dataset
	all_graphs = []
	for shape in shapes:
		edge_indices = torch.load(f'{prj_path}/datasets/edgeindex_{shape}.pt')
		# print(f'edge_indices.shape: {edge_indices.shape}')
		all_nodes = torch.from_numpy(np.load(f'{prj_path}/datasets/features_{shape}.npy')).to(torch.float32)
		# average every node has 2500 features, reduce the number of features to 2500/average_rate using the average of every average_rate features
		all_nodes = torch.mean(all_nodes.reshape(all_nodes.shape[0], -1, average_n_features), dim=2)

		shape_labels = torch.from_numpy(np.load(f'{prj_path}/datasets/targets_{shape}.npy'))
		# construct a graph for each node
		for graph_idx in range(all_nodes.shape[0]):
			graph_edges = edge_indices[:, edge_indices[0] == graph_idx]
			# get graph nodes from the edge indices
			graph_nodes = torch.unique(graph_edges)
			# graph_nodes_map
			graph_nodes_dict = {int(key): value for value, key in enumerate(graph_nodes)}
			# replace each value in graph_edges with the corresponding value in graph_nodes_dict
			new_graph_edges = torch.tensor([[graph_nodes_dict[int(edge[0])], graph_nodes_dict[int(edge[1])]] for edge in graph_edges.T]).T
			# get graph labels
			graph_label = torch.tensor([shape_labels[graph_idx]]).to(torch.float32)

			# construct a graph
			graph = torch_geometric.data.Data(x=all_nodes[graph_nodes], edge_index=new_graph_edges, y=graph_label)
			all_graphs.append(graph)

	# convert to a dataset
	dataset = MyDataset(all_graphs)

	print()
	print(f'Dataset: {dataset}:')
	print('====================')
	print(f'Number of graphs: {len(dataset)}')
	print(f'Number of features: {dataset.num_features}')
	print(f'Number of classes: {dataset.num_classes}')

	data = dataset[0]  # Get the first graph object.

	print()
	print(dataset[0])
	print('graph nodes: ', data.x)
	print('graph edges: ', data.edge_index)
	print(dataset[1])
	print('=============================================================')

	# Gather some statistics about the first graph.
	print(f'Number of nodes: {data.num_nodes}')
	print(f'Number of edges: {data.num_edges}')
	print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
	print(f'Has isolated nodes: {data.has_isolated_nodes()}')
	print(f'Has self-loops: {data.has_self_loops()}')
	print(f'Is undirected: {data.is_undirected()}')

	dataset = dataset.shuffle()
	if train_percent == -1:
		return DataLoader(dataset, batch_size=batch_size)
	else:
		train_dataset = dataset[:int(train_percent * len(dataset))]
		test_dataset = dataset[int(train_percent * len(dataset)):]
		print(f'Number of training graphs: {len(train_dataset)}')
		print(f'Number of test graphs: {len(test_dataset)}')
		train_dl = DataLoader(train_dataset, batch_size=batch_size)
		test_dl = DataLoader(test_dataset, batch_size=batch_size)
		return train_dl, test_dl

class GCN(torch.nn.Module):
	def __init__(self, num_node_features, hidden_channels, num_classes):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(num_node_features, hidden_channels, dtype=torch.float32)
		# self.conv2 = GCNConv(hidden_channels, hidden_channels, dtype=torch.float32)
		# self.conv3 = GCNConv(hidden_channels//2, hidden_channels//4, dtype=torch.float32)
		self.lin = Linear(hidden_channels, num_classes)

	def forward(self, x, edge_index, batch):
		# 1. Obtain node embeddings 
		x = self.conv1(x, edge_index)
		# x = x.relu()
		# x = self.conv2(x, edge_index)
		# x = x.relu()
		# x = self.conv3(x, edge_index)

		# 2. Readout layer
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

		# 3. Apply a final classifier
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin(x)
		
		return x

def test(model, loss_fun, test_dl, device):
	total_loss = 0
	model.eval()
	with torch.no_grad():
		for data in test_dl:
			X = data.x.to(device)
			edge_index = data.edge_index.to(device)
			batch = data.batch.to(device)
			y = data.y.to(device)
			outputs = model(X, edge_index, batch)
			outputs.squeeze_()
			loss = loss_fun(outputs, y)
			total_loss += loss.item()
	avg_loss = total_loss / len(test_dl)
	return avg_loss

def train(model, optimizer, loss_fun, train_dl, device):
	model.train()
	for data in train_dl:
		X = data.x.to(device)
		edge_index = data.edge_index.to(device)
		batch = data.batch.to(device)
		y = data.y.to(device)
		optimizer.zero_grad()
		outputs = model(X, edge_index, batch)
		outputs.squeeze_()
		loss = loss_fun(outputs, y)
		loss.backward()
		optimizer.step()

def train_and_test(model, train_dl, test_dl, optimizer, loss_fun, device, num_epochs, plot_file, detailed_result_file):
	progress_bar = tqdm(range(num_epochs))
	best_weights = None
	best_loss = np.inf
	# plot the train and test losses in each epoch
	train_losses = []
	test_losses = []
	with open(detailed_result_file, 'w') as f:
		print(f'epoch,train_loss,test_loss', file=f)
		overfitting_counter = 0
		for epoch in range(1, num_epochs + 1):
			train(model, optimizer, loss_fun, train_dl, device)
			train_loss = test(model, loss_fun, train_dl, device)
			test_loss = test(model, loss_fun, test_dl, device)
			train_losses.append(train_loss)
			test_losses.append(test_loss)
			print(f'{epoch:5d},{train_loss:20.8f},{test_loss:20.8f}', file=f)
			progress_bar.set_description(f'Epoch {epoch}, Test Loss: {test_loss:.8f}, Train Loss: {train_loss:.8f}')
			progress_bar.update(1)
			# save the best model weights
			if epoch == 1 or test_loss < best_loss:
				best_loss = test_loss
				# deep copy the model weights
				best_weights = copy.deepcopy(model.state_dict())

			# # break if overfitting (if test loss increases for n consecutive epochs)
			# n = 10
			# if epoch > 10 and test_losses[-1] >= test_losses[-2]:
			# 	overfitting_counter += 1
			# 	if overfitting_counter == n:
			# 		print(f'Overfitting detected, breaking at epoch {epoch}')
			# 		break
			# else:
			# 	overfitting_counter = 0 
			
	progress_bar.close()
	# plot the train and test losses with red and blue colors respectively
	plt.plot(range(1, len(train_losses) + 1), train_losses, 'r', label='Train Loss')
	plt.plot(range(1, len(test_losses) + 1), test_losses, 'b', label='Test Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(plot_file)
	# clear the plot
	plt.clf()
	return best_weights, best_loss

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='GNN classification')
	parser.add_argument('--train_percent', type=float, default=0.3, help='Percentage of data to use for training')
	# batch size
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
	# number of epochs
	parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
	# load saved models
	parser.add_argument('--load_saved_models', type=bool, default=False, help='Load saved models')
	# learning rates
	parser.add_argument('--lrs', type=str, default="[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]", help='Learning rates')
	# average n features
	parser.add_argument('--average_n_features', type=int, default=25, help='Average n features')

	train_percent = parser.parse_args().train_percent
	num_epochs = parser.parse_args().num_epochs
	load_saved_models = parser.parse_args().load_saved_models
	batch_size = parser.parse_args().batch_size
	average_n_features = parser.parse_args().average_n_features
	model_path = f'{prj_path}/models/GNN/regression/best_model_{average_n_features}_{train_percent}.pt'
	result_path = f'{prj_path}/results/GNN/regression/regression_{average_n_features}_{train_percent}.csv'
	# create folders if they don't exist
	if not os.path.exists(f'{prj_path}/models/GNN/regression/'):
		os.makedirs(f'{prj_path}/models/GNN/regression/')
	if not os.path.exists(f'{prj_path}/results/GNN/regression/'):
		os.makedirs(f'{prj_path}/results/GNN/regression/')
	if train_percent == -1:
		train_shapes = [
			'euc', 
			'hyperboloid', 
			'poincare', 
			'S2', 
			'S5',
			'sigma.001',
			'sigma.03',
		]
		test_shapes = [
			'S3', 
			'S7',
			'sigma.01',
			'sigma.003',
			'torus'
		]
		print(f'{"-"*20} Using train shapes: {train_shapes} {"-"*20}')
		print(f'{"-"*20} Using test shapes: {test_shapes} {"-"*20}')
		train_dl = prepare_datasets(train_shapes, train_percent, average_n_features, batch_size)
		test_dl = prepare_datasets(test_shapes, train_percent, average_n_features, batch_size)
	else:
		shapes = [
			'euc', 
			'hyperboloid', 
			'poincare', 
			'S2', 
			'S3', 
			'S5', 
			'S7', 
			'sigma.01', 
			'sigma.001',
			'sigma.03',
			'sigma.003', 
			'torus'
		]
		print(f'{"-"*20} Training using {train_percent*100}% of all data {"-"*20}')
		train_dl, test_dl = prepare_datasets(shapes, train_percent, average_n_features, batch_size)

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	print(f'{"-"*20} Using device: {device} {"-"*20}')
	
	model = GCN(num_node_features=2500//average_n_features, hidden_channels=256, num_classes=1)
	initial_weights = copy.deepcopy(model.state_dict())
	loss_fun = nn.MSELoss()

	lrs = eval(parser.parse_args().lrs)

	if not load_saved_models:
		best_weights = None
		best_loss = np.inf
		best_lr = 0
		with open(result_path, 'w') as f:
			print(f'lr,loss', file=f)
			for idx, lr in enumerate(lrs):
				if device.type == 'cuda':
					torch.cuda.empty_cache()
				model.load_state_dict(initial_weights)
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
				model = model.to(device)
				plot_file = f'{prj_path}/results/GNN/regression/loss_{train_percent}_{lr}_{average_n_features}.png'
				detailed_result_file = f'{prj_path}/results/GNN/regression/detailed_result_{train_percent}_{lr}_{average_n_features}.csv'
				weights, loss = train_and_test(model, train_dl, test_dl, optimizer, loss_fun, device, num_epochs, plot_file, detailed_result_file)
				print(f'Loss for lr={lr}: {loss:.8f}')
				print(f'{lr:10.6f},{loss:10.8f}', file=f)
				if idx == 0 or loss < best_loss:
					best_loss = loss
					best_weights = weights
					best_lr = lr

		best_model = model
		best_model.load_state_dict(best_weights)
		print(f'Best loss for lr={best_lr}: {best_loss:.8f}')
		torch.save(best_weights, model_path)
	else:
		# load the saved model
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)
		test_loss = test(model, loss_fun, test_dl, device)
		best_model = model
		print(f'Best model test loss: {test_loss:.8f}')
		train_loss = test(model, loss_fun, train_dl, device)
		print(f'Best model train loss: {train_loss:.8f}')

	# test 20 random samples using the best model and print the results
	random_indices = np.random.choice(len(test_dl), 20)
	for i in random_indices:
		# print in one line
		data = test_dl.dataset[i]
		X = data.x.to(device)
		edge_index = data.edge_index.to(device)
		batch = data.batch
		y = data.y.to(device)
		outputs = best_model(X, edge_index, batch)
		print(f'predicted: {outputs.item():10.4f}, golden: {y.item():10.4f}')
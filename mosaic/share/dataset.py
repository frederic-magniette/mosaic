import torch
import numpy as np

class dataset_logical_or():
	def __init__(self):
		...
	
	def prepare(self, data, run_info, module_info):
		# Data == None, we don't need it in this example beacause dataset()
		# is the first class in the pipeline process

		X = []
		y = []

		# data_size found in configuration file
		data_size = int(module_info['data_size'])

		for i in range(data_size):
			arr = np.random.randint(2, size=8)
			X.append(arr)
			y.append(arr[:4] | arr[4:])

		X = torch.tensor(X).float()
		y = torch.tensor(y).float()
		dataset = torch.utils.data.TensorDataset(X, y)


		# train proportion is also in configuration file
		train_prop = float(module_info['train_prop'])
		train_size = int(len(dataset) * train_prop)
		test_size = len(dataset) - train_size

		train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

		self.batch_size = int(module_info['batch_size'])

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)

		return train_loader, test_loader


	def info(self):
		return {'batch_size' : self.batch_size, 'input_size' : 8, 'output_size': 4}


class dataset_logical_xor():
	def __init__(self):
		...
	
	def prepare(self, data, run_info, module_info):
		# Data == None, we don't need it in this example beacause dataset()
		# is the first class in the pipeline process

		X = []
		y = []

		# data_size found in configuration file
		data_size = int(module_info['data_size'])

		for i in range(data_size):
			arr = np.random.randint(2, size=8)
			X.append(arr)
			y.append(arr[:4] ^ arr[4:])

		X = torch.tensor(X).float()
		y = torch.tensor(y).float()
		dataset = torch.utils.data.TensorDataset(X, y)


		# train proportion is also in configuration file
		train_prop = float(module_info['train_prop'])
		train_size = int(len(dataset) * train_prop)
		test_size = len(dataset) - train_size

		train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

		self.batch_size = int(module_info['batch_size'])

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)

		return train_loader, test_loader


	def info(self):
		return {'batch_size' : self.batch_size, 'input_size' : 8, 'output_size': 4}

class dataset_OR():
	def __init__(self):
		...
	
	def prepare(self, data, run_info, module_info):

		data_size = int(module_info['data_size'])
		X = np.array([np.random.randint(2, size=2) for _ in range(data_size)])
		y = np.array([a | b for a, b in X])

		X = torch.tensor(X).float()
		y = torch.tensor(y).float()
		dataset = torch.utils.data.TensorDataset(X, y)


		# train proportion is in configuration file

		train_prop = float(module_info['train_prop'])
		train_size = int(len(dataset) * train_prop)
		test_size = len(dataset) - train_size

		# Retirer le ransom_split ?
		train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

		# batch size is also in configuration file
		self.batch_size = int(module_info['batch_size'])

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
							  drop_last=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
							  drop_last=True)

		return train_loader, test_loader


	def info(self):
		#	Here we must return batch_size because it used in the run part
		#	We'll also need input_size and output_size in ours mlp

		return {'batch_size' : self.batch_size, 'input_size' : 2, 'output_size': 1}
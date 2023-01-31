import torch
import numpy as np

class mlp_funnel(torch.nn.Module):
	def __init__(self, input_size, output_size, length, width):
		super(mlp_funnel, self).__init__()
		self.length = int(length)
		self.width = int(width)
		self.input_size = int(input_size)
		self.output_size = int(output_size)
		self.layers = torch.nn.ModuleList()
		self.activation = torch.nn.Sigmoid()
		sizes = list(self._calc_sizes())
		sizes = [self.input_size] + sizes + [self.output_size]

		for i in range(len(sizes) - 1):
				self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

	def _calc_sizes(self):
		sizes = np.full(fill_value=self.width, shape=self.length)
		ret = 0
		for i in range(len(sizes) - 1):
			ret = sizes[i] // 2
			sizes[i + 1] += sizes[i] - ret
			sizes[i] //= 2
		return sizes[::-1]

	def forward(self, data, run_info, module_info):
		for layer in self.layers:
			data = self.activation(layer(data))
		return data

	def info(self):
		return {}

	def save_model(self, path):
		path += '.pt'
		torch.save(self, path)
	
	def load_model(self, path):
		path += '.pt'
		self = torch.load(path)


class mlp_brick(torch.nn.Module):
	def __init__(self, input_size, output_size, length, width):
		super(mlp_brick, self).__init__()
		self.length = int(length)
		self.width = int(width)
		self.input_size = int(input_size)
		self.output_size = int(output_size)
		self.layers = torch.nn.ModuleList()
		self.activation = torch.nn.Sigmoid()
		sizes = [self.input_size] + [self.width for _ in range(self.length)] + [self.output_size]

		for i in range(len(sizes) - 1):
				self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
	
	def forward(self, data, run_info, module_info):
		for layer in self.layers:
			data = self.activation(layer(data))
		return data

	def info(self):
		return {}

	def save_model(self, path):
		path += '.pt'
		torch.save(self, path)
	
	def load_model(self, path):
		path += '.pt'
		self = torch.load(path)

from torch.nn import MSELoss


class Mse(MSELoss):
	def __init__(self):
		super(Mse, self).__init__()
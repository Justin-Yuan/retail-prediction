# solver to optimize for the model 

import os 
import numpy as np
import pandas as pd 
import pickle 

import torch 
import torchvision 
import torch.utils.data 
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable 

from model import RetailModel


class Solver(object):
	""" build the model, trains it with the data being loaded and infer on new data 
	"""
	def __init__(self, config, model_dims, data_loader, reuse=False):
		self.lr = config.learning_rate
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size
		self.log_step = config.log_step
		self.model_dir = config.model_dir 

		self.build_model(model_dims, reuse)
		self.data_loader = data_loader

	def build_model(self, dims, reuse=False, param_path=None):
		self.model = RetailModel(dims)
		# the weight decay argument for L2 regularization 
		self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=0.005)
		self.loss = nn.MSELoss()
		if reuse:
			self.model.load_state_dict(torch.load(param_path))


	def to_variable(self, x):
		"""Convert tensor to variable."""
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x)

	def train(self):
		""" start training on the model 
		"""
		total_step = len(self.data_loader)
		for epoch in range(self.num_epochs):
			for i, (x, y) in enumerate(self.data_loader):
				x  = self.to_variable(x.float())
				y = self.to_variable(y.float())

				self.optimizer.zero_grad()
				output = self.model(x)
				loss = self.loss(output, y)
				loss.backward()
				self.optimizer.step()

				if (i+1) % self.log_step == 0:
					print( 'Epoch [%d/%d], Step[%d/%d], loss: %.4f, ' 
                          % (epoch+1, self.num_epochs, i+1, total_step, loss.data[0]))


			# save the model per epoch, only save parameters 
			model_path = os.path.join(self.model_dir, 'model-%d.pkl' %(epoch+1))
			torch.save(self.model.state_dict(), model_path)


	def inference(self, x):
		""" make predictions on test data 

		Input
			x: a numpy array 
		Output 
			a numpy array as output 
		"""
		x = self.to_variable(torch.from_numpy(x))
		output = self.model(x)
		print("Output: ", output.data)
		return output.data.numpy()










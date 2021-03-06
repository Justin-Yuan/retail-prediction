# model file 

import numpy as np
import pandas as pd 
import pickle 

import torch 
import torchvision 
import torch.utils.data 
import torch.nn as nn



class RetailModel(nn.Module):
	""" feed-forward neural nets  
	"""
	def __init__(self, dims=[(8, 100), (100, 20), (20, 5), (5, 1)]):
		super(RetailModel, self).__init__()

		self.dims = dims 
		self.fcs = []
		for i in range(len(self.dims)):
			in_dim, out_dim = self.dims[i]
			self.fcs.append(nn.Linear(in_dim, out_dim))
			# self.fcs.append(nn.ReLU())
			self.fcs.append(nn.ELU())
		self.net = nn.Sequential(*self.fcs)

	def forward(self, x):
		# output = x
		# for i in range(len(self.dims)):
		# 	output = self.relu(self.fcs[i](output))
		return self.net(x)


class Ensemble(nn.Module):
	""" an ensemble of models trained for 3 years

	Input
		ensembles: list of models  
	"""
	def __init__(self, ensembles):
		super(Ensemble, self).__init__()
		self.ensembles = ensembles


	def forward(self, x):
		 # there should be at least one model in the ensembles list 
		 output = self.ensembles[0](x)
		 for model in self.ensembles[1:]:
		 	output += model(x)
		 output = output / len(self.ensembles)
		 return output 













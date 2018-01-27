# preprocess data 

import os
import numpy as np 
import pandas as pd 
import pickle 
from torch.utils import data
from torchvision import transforms


class RetailDataset(data.Dataset):
	""" build customized dataset 
	"""
	def __init__(self, path, mode="train", transform=False):
		with open(path, 'rb') as f:
			data = pickle.load(f)
		self.features = []
		self.X = data[mode]["X"]
		self.Y = data[mode]["Y"]

	def __getitem__(self, index):
		x = self.X[index]
		y = self.Y[index]
		return x, y

	def __len__(self):
		assert len(self.X) == len(self.Y)
		return len(self.Y)



def get_loader(path, batch_size, num_workers):
	dataset = RetailDataset(path)
	data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
	return data_loader
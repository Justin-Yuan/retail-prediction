# main execution file 

import numpy as np 
import argparse
import pickle 
import torch
import torch.nn as nn 
from torch.autograd import Variable

from data_loader import RetailDataset, get_loader
from solver import Solver 
from model import RetailModel, Ensemble 


def main(config):
	# main execution file

	data_loader_train = get_loader(path = config.data_path, 
							 batch_size = config.batch_size, 
							 num_workers = config.num_workers,
							 mode="train")
	data_loader_valid = get_loader(path = config.data_path, 
							 batch_size = config.batch_size, 
							 num_workers = config.num_workers,
							 mode="validation")
	data_loader_test = get_loader(path = config.data_path, 
							 batch_size = config.batch_size, 
							 num_workers = config.num_workers,
							 mode="test")
	data_loader = {"train": data_loader_train,
				   "validation": data_loader_valid,
				   "test": data_loader_test
				   }
	model_dims = [(config.input_dim, 200), (200, 100), (100, 50), (50, 10), (10, 1)]

	if config.mode == "train":
		# for model training phase 
		solver = Solver(config, model_dims, data_loader)
		solver.train()
	elif config.mode == "validation":
		# for model validation phase 
		solver = Solver(config, model_dims, data_loader, reuse=True, param_path=config.model_path)
		solver.validation()
	elif config.mode == "test":
		# for model test phase 
		solver = Solver(config, model_dims, data_loader, reuse=True, param_path=config.model_path)
		solver.test()
	elif config.mode == "ensemble":
		# for model ensemble evaluation 
		path0 = "models/model-20.pkl"
		net0 = RetailModel(model_dims)
		net0.load_state_dict(torch.load(path0))

		path1 = "models_2009/model-20.pkl"
		net1 = RetailModel(model_dims)
		net1.load_state_dict(torch.load(path1))
		
		path2 = "models_2010/model-20.pkl"
		net2 = RetailModel(model_dims)
		net2.load_state_dict(torch.load(path2))
		
		path3 = "models_2011/model-20.pkl"
		net3 = RetailModel(model_dims)
		net3.load_state_dict(torch.load(path3))

		ensemble = Ensemble([net0, net1, net2, net3])
		l = nn.MSELoss()

		# evaluate loss 
		for loader in [data_loader_train, data_loader_valid, data_loader_test]:
			x, y = next(iter(loader))
			x = Variable(x.float())
			y = Variable(y.float())
			output = ensemble(x)
			loss = l(output, y)
			print( 'Loss with Ensembles: %.4f, ' % (loss.data[0]))
	else:
		# other invalid modes 
		print("invalid mode")






if __name__ == "__main__":
	parser = argparse.ArgumentParser() 

	# hyperparameters
	parser.add_argument("--data_path", type=str, default="../data/datasets.pkl")
	parser.add_argument("--model_dir", type=str, default="./models")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=0.0001)
	parser.add_argument("--log_step", type=int, default=100)
	parser.add_argument("--input_dim", type=int, default=175)
	parser.add_argument("--mode", type=str, default="train")
	parser.add_argument("--model_path", type=str, default="./models/model-6.pkl")


	config = parser.parse_args()
	print(config)
	main(config)

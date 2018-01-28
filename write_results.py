import pandas as pd
import numpy as np
import pickle

file_name = "data/hackathon_result_old.dat"
res = pd.read_csv(file_name)
# print (res)
results_name = "results/predictions.pkl"

with open(results_name, "rb") as f:
	quantity = pickle.load(f)


res["quantity"] = quantity



res.to_csv("results/hackathon_result.dat")

res = pd.read_csv("results/hackathon_result.dat")
print(res.head())

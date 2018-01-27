import numpy as np

myarray = np.fromfile('hackathon_dataset_2009.dat',dtype=float)
# print (myarray.shape)
# np.savetxt()

# Date, store, Department ID, Item ID, unit price, quanitity, on_promotijon flag (Y or N), promo_type ID

data_filenames = ['hackathon_dataset_2009.dat', 'hackathon_dataset_2010.dat', 'hackathon_dataset_2011.dat']


for filename in data_filenames:
	with open(filename) as f:
		content = f.readlines()

		n_points = len(content)
		print (n_points)
		# print (content)

		# lines = content.split(',')
		# for line in content:

		# 	data = line.split(',')
		# 	date = data[0]
		# 	store = data[1]
		# 	department_ID = data[2]
		# 	item_ID = data[3]
		# 	unit_price = data[4]
		# 	quantity = data[5]
		# 	on_promotion = int(data[6] == 'Y')
		# 	promo_type = data[7].replace('\n', '')






import numpy as np

datapath = "data/new"
datafilelist = ["roundv2-2.npy","roundv2-3.npy"]
arr1 = np.load(datapath+"/"+datafilelist[0], allow_pickle=True)
shape = arr1.shape[1]
arr1 = arr1.flatten()

arr2 = np.load(datapath+"/"+datafilelist[1], allow_pickle=True)
arr2 = arr2.flatten()
arr3 = np.append(arr1, arr2)

arr3 = arr3.reshape(arr3.shape[0]//12,12)
print(arr3.shape)
filename = "data/new/wrongrecord.npy"
np.save(filename, arr3)
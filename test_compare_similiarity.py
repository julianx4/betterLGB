import numpy as np

datapath = "data/new"
datafilelist = ["roundv2-2.npy","roundv2-3.npy","roundv2-4.npy","roundv2-5.npy","roundv2-6.npy","roundv2-17.npy","wrongrecord.npy"]
arr = np.load(datapath+"/"+datafilelist[0], allow_pickle=True)
arr = arr[:, [10]]
arr = arr.flatten()

for file in datafilelist:
    filewithpath = datapath+"/"+file
    prev_arr = arr
    
    arr = np.load(filewithpath, allow_pickle=True)
    arr = arr[:, [10]]
    arr = arr.flatten()

    if len(arr) < len(prev_arr):
        new_length = min(len(arr), len(prev_arr))
        indices_new = np.linspace(0, len(prev_arr) - 1, new_length)
        prev_arr = np.interp(indices_new, np.arange(len(prev_arr)), prev_arr)

    elif len(arr) > len(prev_arr):
        new_length = min(len(arr), len(prev_arr))
        indices_new = np.linspace(0, len(arr) - 1, new_length)
        arr = np.interp(indices_new, np.arange(len(arr)), arr)

    cos_sim = np.dot(prev_arr, arr) / (np.linalg.norm(prev_arr) * np.linalg.norm(arr))

    print(cos_sim)
    #print(arr)

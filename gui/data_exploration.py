# IPython log file

import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

data_red = io.readsav('test_data_24apr11.010_red.sav')
data_cal = io.readsav('test_data_24apr11.010_cal.sav')

data_cal.keys()
#[Out]# dict_keys(['disp', 'lambda_ini', 'l1', 'l2', 'lambda'])

for key in data_cal.keys():
    print(key, type(data_cal[key]))
    if type(data_cal[key]) == np.ndarray:
        print(key, data_cal[key].shape)

for key in data_red.keys():
    print(key, type(data_red[key]))
    if type(data_red[key]) == np.ndarray:
        print(key, data_red[key].shape)

data_cube = data_red['st']
frequencies = data_cal['lambda']
data_save = {'lambda': frequencies,}
data_save['data'] = data_cube
data_save.keys()
datos = [data_save]
datos[0]['data'] = np.moveaxis(datos[0]['data'], 1, 0)
datos[0]['data'].shape
#[Out]# (60, 4, 223, 165)

with open("data.pkl", "wb") as output_file:
    pkl.dump(datos, output_file)
    
exit()

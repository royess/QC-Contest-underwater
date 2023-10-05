import numpy as np
import pandas as pd

data=pd.read_pickle('./NoiseModel/fakecairo.pkl')
f = open('./NoiseModel/fakecairo_txt_1.txt','w')
f.write(str(data))
f.close()
f = open('./NoiseModel/fakecairo_structure.txt','w')
f.write(f'length of ["errors"]: {len(data["errors"])}')
f.write('\n')
for i in range(len(data['errors'])):
    f.write(str(data['errors'][i].keys()))
    f.write('\n')
f.close()
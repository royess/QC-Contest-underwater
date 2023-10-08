import numpy as np
import pandas as pd

data=pd.read_pickle('./NoiseModel/fakemontreal.pkl')
f = open('./NoiseModel/fakemontreal.txt','w')
f.write(str(data))
f.close()
# f = open('./NoiseModel/fakekolkata_structure.txt','w')
# f.write(f'length of ["errors"]: {len(data["errors"])}')
# f.write('\n')
# for i in range(len(data['errors'])):
#     f.write(str(data['errors'][i].keys()))
#     f.write('\n')
# f.close()
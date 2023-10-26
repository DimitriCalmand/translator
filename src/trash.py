import pandas as pd
import numpy as np
df = pd.read_csv("../data/translate.csv", chunksize = 4)
for i in df:
    print(np.array(i['en']))
    break

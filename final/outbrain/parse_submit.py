import pandas as pd
import numpy as np

dtypes = {'clicked': np.float32}
df = pd.read_csv("proba.csv", dtype=dtypes)
df.sort_values(['display_id','clicked'], inplace=True, ascending=False)
subm = df.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
subm.to_csv("subm.csv", index=False)

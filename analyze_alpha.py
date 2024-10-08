import pandas as pd
import numpy as np

df = pd.read_csv("out/output.csv")
df["ratio_gt2_d2"] = df.apply(lambda row: row["gt2"] / row["d2"], axis=1)
#
df["ratio_gt2_d2"] = df["ratio_gt2_d2"].round(4)
# df["error_1"] = df["error_1"].round(1)
# df["error_2"] = df["error_2"].round(1)
# df["error_3"] = df["error_3"].round(1)

df['short'] =df['imagename'].str.slice(-6,-1)
X=df[['short', 'gt1', 'gt2', 'gt3', 'w1', 'w2', 'w3','error1', 'error2', 'error3']]


find_large = lambda df, error: df[(df['error1'].abs() > error) | (df['error2'].abs() > error) | (df['error3'].abs() > error)]


# find either error1 or error2 or error3 is larger than 20%
find_large(df, 20)


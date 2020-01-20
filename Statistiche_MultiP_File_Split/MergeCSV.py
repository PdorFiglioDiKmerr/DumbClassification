import pandas as pd
import os

def merge_csv (directory):

    dataset_final = pd.DataFrame()
    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' in file:
                df_app = pd.read_csv(os.path.join(r, file))
                dataset_final = pd.concat([dataset_final, df_app])

    dataset_final.to_csv( os.path.join(directory, "dataset.csv") )
    return dataset_final

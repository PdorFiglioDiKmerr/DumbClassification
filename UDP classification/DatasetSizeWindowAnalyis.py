import pandas as pd
import os
import matplotlib.pyplot as plt
from json import JSONDecoder, JSONDecodeError
import sys
import numpy as np
from pcap_manager import pcap_manager
import re
import warnings
from utilities import save_photo
warnings.filterwarnings("ignore")

data = {}
train_dir = "/home/det_tesi/sgarofalo/Window size analysis/train_dataset"
save_dir = "/home/det_tesi/sgarofalo/Window size analysis"
for seconds in range(1, 11):
    seconds_samples = str(seconds) + "s"
    pm = pcap_manager(seconds_samples)
    print("Building datasets with seconds_samples = " + seconds_samples)
    pm.merge_pcap(train_dir)
    dataset = pd.read_csv("/home/det_tesi/sgarofalo/Window size analysis/train_dataset/dataset.csv")
    data[seconds] = len(dataset)
    print()
df = pd.DataFrame(columns=["Size"])
for i in data:
    df = df.append({"Size": data[i]}, ignore_index=True)
df.index +=  1
plt.figure(figsize=(16, 9))
plt.plot(df)
plt.xlabel("Seconds")
plt.ylabel("Size")
t = "Dataset size analysis"
plt.title(t, fontsize=20)
plt.grid()
plt.tight_layout()
save_photo(save_dir, t)
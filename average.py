#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


filenames = glob.glob("data1.txt")
filenames = sorted(filenames, key=os.path.basename)

for filename in filenames:
    df = pd.read_table(filename, header = None, sep = r"\s+")
    data = df.iloc[:,0]
    aver = np.float(0.0)
    
    for i in data:
        aver += i
    aver /= len(data)
    print(filename, aver)


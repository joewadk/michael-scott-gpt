import numpy as np
import pandas as pd
import os
file_path = 'data/talking_head.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
df_filtered = df[df['character'] == 'Michael']
# Save the filtered dataframe to a CSV file
file_name = 'clean.csv'
df_filtered.to_csv(file_name, index=False)

import os
import chardet
import pandas as pd

default_chosen_security_file_path = 'Dissertation Project 1\Data\K线导出_600873_日线数据.xlsx'
df = pd.read_excel(default_chosen_security_file_path)
print(df.head())
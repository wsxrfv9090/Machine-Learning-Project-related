import pandas as pd

# Original file path
input_file_path = r'D:\ImportanFiles\Coding Related\Repositories\Machine Learning project related\Project 2\Data\002796_metadata_gbk.csv'

# Output file path with UTF-8 encoding
output_file_path = r'D:\ImportanFiles\Coding Related\Repositories\Machine Learning project related\Project 2\Data\002796_metadata_utf8.csv'

# Read the original file with its current encoding (assuming GBK encoding here)
df = pd.read_csv(input_file_path, encoding='GBK')

# Write the DataFrame to a new file with UTF-8 encoding
df.to_csv(output_file_path, encoding='utf-8', index=False)

print("File has been successfully converted to UTF-8.")

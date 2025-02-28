import pandas as pd

# Original file path
input_file_path = r'D:\Important Files\Repositories\Machine-Learning-Project-related\Dissertation Project 1\Data\600873_RAW.CSV'

# Output file path with UTF-8 encoding
output_file_path = r'D:\Important Files\Repositories\Machine-Learning-Project-related\Dissertation Project 1\Data\600873_UTF-8.CSV'

# Read the original file with its current encoding (assuming GBK encoding here)
df = pd.read_csv(input_file_path, encoding='GBK')

# Write the DataFrame to a new file with UTF-8 encoding
df.to_csv(output_file_path, encoding='utf-8', index=False)

print("File has been successfully converted to UTF-8.")

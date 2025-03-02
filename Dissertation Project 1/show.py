import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

# Read data from CSV file with 'Stock ID' as a string
file_path = 'D:\Important Files\Repositories\Machine-Learning-Project-related\Dissertation Project 1\Output\Output CSV\output_history.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, dtype={'Stock ID': str})

# Keep only the relevant columns
df = df[['Stock ID', 'Alpha', 'Beta']]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each stock's line using alpha as the intercept and beta as the slope
for _, row in df.iterrows():
    x = np.linspace(0, 10, 100)  # x values for the line
    y = row['Alpha'] + row['Beta'] * x  # y = alpha + beta * x
    
    # Plot the line and annotate with the Stock ID
    plt.plot(x, y, label = f"{row['Stock ID']} (α={row['Alpha']:.2f}, β={row['Beta']:.2f})")
    plt.text(x[-1], y[-1], row['Stock ID'], fontsize=9, verticalalignment='bottom')

# Set title and labels
plt.xlabel('Market Excess Returns (E(R_m) - R_f)')
plt.ylabel('Stock Excess Returns (E(R_i) - R_f)')
plt.title("Stock Lines Based on Alpha and Beta")
plt.xlim(0, 0.01)
plt.ylim(-0.001, 0.015)
plt.legend()
plt.grid(True)

current_dir = os.getcwd()
print(current_dir)
output_directory = os.path.join(current_dir, 'Dissertation Project 1\Output')
output_all_lines_directory = os.path.join(output_directory, 'All Lines')
os.makedirs(output_all_lines_directory, exist_ok=True)
now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_img = os.path.join(output_all_lines_directory, now_str + ".png")
if os.path.exists(output_img):
    os.remove(output_img)
    print(f"Deleted existing file: {output_img}.")
    plt.savefig(output_img)
    print(f"Recreated a new image at: {output_img}.")
else:
    plt.savefig(output_img)
    print(f"Figure Saved to :{output_img}. ")
import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
angle_df = pd.read_csv('BCGAngleOffset.csv')
bcg_df = pd.read_csv('reduced_clusters_locals_main2.csv')

# Optional: display the first few rows of the DataFrame to verify it loaded correctly
print(angle_df.head())
print(bcg_df.head())
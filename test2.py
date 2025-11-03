import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('50deg.csv')

plt.figure(figsize=(8, 6))
plt.scatter(df['coord_ra'], df['coord_dec'], s=10)  # s controls point size
plt.xlabel('RA')
plt.ylabel('Dec')
plt.title('RA vs Dec Scatter Plot')
plt.grid(True)
plt.savefig('50deg.png', dpi=300, bbox_inches='tight')
plt.show()
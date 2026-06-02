import numpy as np

data = np.load("train.npz")

print("Keys:", data.files)
print("T shape:", data['T'].shape)
print("Q shape:", data['Q'].shape)
print("T range:", data['T'].min(), data['T'].max())
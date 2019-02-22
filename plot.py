import pickle
import matplotlib.pyplot as plt

with open('./portfolio_val/201902201627-test.p', 'rb') as f:
    val = pickle.load(f)

plt.plot(val)
plt.show()
import numpy as np

def beer(x, a0, a1, a2, a3, a4):
	fit = a0 \
	+ a1 * np.sin(2 * np.pi * x) \
	+ a2 * np.cos(2 * np.pi * x) \
	+ a3 * np.sin(2 * np.pi * 2 * x) \
	+ a4 * np.cos(2 * np.pi * 2 * x)
	return fit
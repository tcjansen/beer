import numpy as np
import matplotlib.pyplot as plt

def flat(x, a0):
	return np.zeros(len(x)) + (a0 * 1e-6 + 1)

def beer(x, a0, a1, a2, a3, a4):
	fit = a0 \
	+ a1 * np.sin(2 * np.pi * x) \
	+ a2 * np.cos(2 * np.pi * x) \
	+ a3 * np.sin(2 * np.pi * 2 * x) \
	+ a4 * np.cos(2 * np.pi * 2 * x)
	return fit

def therm(x, a0, a1):
	return (a0 * 1e-6 + 1) - 10**a1 * np.cos(x * 2 * np.pi)

def ellip(x, a0, a1):
	return (a0 * 1e-6 + 1) - 10**a1 * np.cos(4 * np.pi * x)

def shifted_therm(x, a0, a1, phi):
	return (a0 * 1e-6 + 1) - 10**a1 * np.cos((x + phi) * 2 * np.pi)

def therm_ellip(x, a0, a1, a2):
	return (a0 * 1e-6 + 1) - 10**a1 * np.cos(x * 2 * np.pi) - 10**a2 * np.cos(4 * np.pi * x)

def therm_beam(x, a0, a1, a2):
	return (a0 * 1e-6 + 1) - 10**a1 * np.cos(x * 2 * np.pi) + 10**a2 * np.sin(2 * np.pi * x)

def ellip_beam(x, a0, a1, a2):
	return (a0 * 1e-6 + 1) - 10**a1 * np.cos(4 * np.pi * x) + 10**a2 * np.sin(2 * np.pi * x)
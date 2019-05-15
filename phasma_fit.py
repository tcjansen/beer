import numpy as np
import mcmc
import os
import models
import scipy.stats as sstats
import matplotlib.pyplot as plt

class Fit():

	def __init__(self, TICID, model=None, nwalkers=100, nsteps=1000, nbins=21):
		self.model = model
		self.id = TICID
		self.nwalkers = nwalkers
		self.nsteps = nsteps
		self.nbins = nbins

		phase, flux, flux_err = np.genfromtxt(os.getcwd() + '/targets/' + str(TICID) \
			+ '/' + str(TICID) + '_phasma_final.csv', delimiter=',', usecols=(2,3,4), \
			unpack=True)
		period, t14 = np.genfromtxt(os.getcwd() + '/targets/' + str(TICID) + \
			'/target_info.csv', delimiter=',', usecols=(1,2), unpack=True)
		t14 /= 24 #hours to days

		#remove the transit
		intransit = (phase > -1.05*t14/period/2) & (phase < 1.05*t14/period/2)
		phase[intransit] = np.nan
		flux[intransit]= np.nan
		flux_err[intransit]= np.nan

		# bin the data
		bins = np.linspace(-0.5, 0.5, nbins)
		digitized = np.digitize(phase, bins)

		bin_phase = bins + (bins[1] - bins[0]) / 2
		bin_phase = bin_phase[:-1]

		# bin_flux = np.array([np.average(flux[digitized == i], weights=1/flux_err[digitized == i]) for i in range(1, len(bins))])
		bin_flux = np.array([flux[digitized==i].mean() for i in range(1, len(bins))])
		bin_flux_err = np.array([flux_err[digitized == i].mean() \
			for i in range(1, len(bins))]) / np.sqrt(len(phase)/nbins)
		
		intransit = (bin_phase > -1.05*t14/period/2)&(bin_phase < 1.05*t14/period/2)
		bin_phase = bin_phase[~intransit]
		bin_flux = bin_flux[~intransit]
		bin_flux_err = bin_flux_err[~intransit]

		self.phase = phase
		self.flux = flux
		self.flux_err = flux_err
		self.bin_phase = bin_phase
		self.bin_flux = bin_flux
		self.bin_flux_err = bin_flux_err

	def get_params(self, model=None):
		if model=="flat":
			return ['offset - 1 [ppm]'], 1
		elif model=="therm":
			return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$'], 2
		elif model=="ellip":
			return ['offset - 1 [ppm]', r'log10$A_{ellip}$'], 2
		elif model=="shifted_therm":
			return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$', r'$\phi$'], 3
		elif model=="therm_ellip":
			return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$', r'log10$A_{ellip}$'], 3
		elif model=="therm_beam":
			return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$', r'log10$A_{beam}}$'], 3
		elif model=="ellip_beam":
			return ['offset - 1 [ppm]', r'log10$A_{ellip}$', r'log10$A_{beam}$'], 3
		else:
			raise ValueError("The correct model type must be specified when calling for the model parameters")

	def get_pos(self, theta, model=None):
		if model=="flat":
			return [np.array([theta[0]+1e-2*np.random.randn()]) for i in range(self.nwalkers)]
		elif model=="therm":
			return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn()])
				   for i in range(self.nwalkers)]
		elif model=="ellip":
			return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn()])
				   for i in range(self.nwalkers)]
		elif model=="shifted_therm":
			return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(), 
					theta[2]+1e-2*np.random.randn()]) 
					for i in range(self.nwalkers)]
		elif model=="therm_ellip":
			return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(), 
					theta[2] - 0.1 * np.random.randn()]) 
					for i in range(self.nwalkers)]
		elif model=="therm_beam":
			return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(), 
					theta[2] - 0.1 * np.random.randn()]) 
					for i in range(self.nwalkers)]
		elif model=="ellip_beam":
			return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(), 
					theta[2] - 0.1 * np.random.randn()]) 
					for i in range(self.nwalkers)]
		else:
			raise ValueError("The correct model type must be specified when calling for the walker starting positions")

	def results(self, theta, priors, convergenceplot_name=None, cornerplot_name=None):

		params, ndim = self.get_params(model=self.model)
		pos = self.get_pos(theta, model=self.model)
		samples = mcmc.get_samples(self.id, params, pos, ndim, self.nwalkers, \
			self.nsteps, theta, getattr(models, self.model), priors, \
			self.phase, self.flux, self.flux_err, \
			convergenceplot_name=convergenceplot_name, \
			cornerplot_name=cornerplot_name)

		return samples

	def plot(self, coefficients, fits=None, plot_name=None, one_plot=False):

		plotting_phase = np.linspace(-0.5, 0.5, len(self.phase))
		plt.figure(figsize=(8,4))
		plt.errorbar(self.bin_phase, (self.bin_flux - 1)*1e6, \
			yerr=self.bin_flux_err*1e6, fmt='o', color='#E71D36')

		if one_plot:

			for i in range(len(fits)):
				if fits[i]=="flat":
					plt.plot(plotting_phase, (models.flat(plotting_phase, \
						coefficients[i][0]) - 1)*1e6, lw=2, color='#2EC4B6', ls=None)
				elif fits[i]=="beer":
					plt.plot(plotting_phase, (models.beer(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2], \
						coefficients[i][3], coefficients[i][4]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='--', label="beer")
				elif fits[i]=="therm":
					plt.plot(plotting_phase, (models.therm(plotting_phase, \
						coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls=None, label="therm + ref")
				elif fits[i]=="ellip":
					plt.plot(plotting_phase, (models.ellip(plotting_phase, \
						coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls=None, label="ellip")
				elif fits[i]=="shifttherm":
					plt.plot(plotting_phase, (models.shifted_therm(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="shifted therm + ref")
				elif fits[i]=="thermellip":
					plt.plot(plotting_phase, (models.therm_ellip(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="therm + ellip")
				elif fits[i]=="thermbeam":
					plt.plot(plotting_phase, (models.therm_beam(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="therm + beam")
				elif fits[i]=="ellipbeam":
					plt.plot(plotting_phase, (models.ellip_beam(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="ellip + beam")
				plt.ylabel('ppm')
				plt.legend()

		else:

			if len(fits)==1:

				if fits[i]=="flat":
					plt.plot(plotting_phase, (models.flat(plotting_phase, \
							 coefficients[i][0]) - 1)*1e6, lw=2, color='#2EC4B6', ls=None)
				elif fits[i]=="beer":
					plt.plot(plotting_phase, (models.beer(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2], \
						coefficients[i][3], coefficients[i][4]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='--', label="beer")
				elif fits[i]=="therm":
					plt.plot(plotting_phase, (models.therm(plotting_phase, \
						coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls=None, label="therm + ref")
				elif fits[i]=="ellip":
					plt.plot(plotting_phase, (models.ellip(plotting_phase, \
						coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls=None, label="ellip")
				elif fits[i]=="shifttherm":
					plt.plot(plotting_phase, (models.shifted_therm(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="shifted therm + ref")
				elif fits[i]=="thermellip":
					plt.plot(plotting_phase, (models.therm_ellip(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="therm + ellip")
				elif fits[i]=="thermbeam":
					plt.plot(plotting_phase, (models.therm_beam(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="therm + beam")
				elif fits[i]=="ellipbeam":
					plt.plot(plotting_phase, (models.ellip_beam(plotting_phase, \
						coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
						color='#FF9F1C', lw=2, ls='-.', label="ellip + beam")
				plt.xlabel('orbital periods since transit')

			else:

				fig, axs = plt.subplots(len(fits), 1, sharex=True)
				for i in range(len(fits)):
					if fits[i]=="flat":
						axs[i].plot(plotting_phase, (models.flat(plotting_phase, \
							coefficients[i][0]) - 1)*1e6, lw=2, color='#FF9F1C')
						axs[i].set_title("flat model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="therm":
						axs[i].plot(plotting_phase, (models.therm(plotting_phase, \
							coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="therm + ref")
						axs[i].set_title("therm + ref model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="ellip":
						axs[i].plot(plotting_phase, (models.ellip(plotting_phase, \
							coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="ellip")
						axs[i].set_title("ellip model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="shifttherm":
						axs[i].plot(plotting_phase, (models.shifted_therm(plotting_phase, \
							coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="shifted therm + ref")
						axs[i].set_title("shifted therm + ref model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="beer":
						axs[i].plot(plotting_phase, (models.beer(plotting_phase, \
							coefficients[i][0], coefficients[i][1], coefficients[i][2], \
							coefficients[i][3], coefficients[i][4]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="beer")
						axs[i].set_title("beer model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="thermellip":
						axs[i].plot(plotting_phase, (models.therm_ellip(plotting_phase, \
							coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="therm + ellip")
						axs[i].set_title("therm + ellip model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="thermbeam":
						axs[i].plot(plotting_phase, (models.therm_beam(plotting_phase, \
							coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="therm + beam")
						axs[i].set_title("therm + beam model")
						axs[i].set_ylabel("[ppm]")

					elif fits[i]=="ellipbeam":
						axs[i].plot(plotting_phase, (models.ellip_beam(plotting_phase, \
							coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
							color='#FF9F1C', lw=2, label="ellip + beam")
						axs[i].set_title("ellip + beam model")
						axs[i].set_ylabel("[ppm]")

					axs[i].errorbar(self.bin_phase, (self.bin_flux - 1)*1e6, \
							 yerr=self.bin_flux_err*1e6, fmt='o', color='#E71D36')

				axs[len(fits) - 1].set_xlabel('orbital periods since time of inferior conjunction')
				fig.set_size_inches(6, 2*len(fits))

		plt.xlim(-0.5, 0.5)
		plt.savefig(os.getcwd() + '/targets/' + str(self.id) + '/' + plot_name)

		return

			

		
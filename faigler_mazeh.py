import numpy as np
import astropy.modeling.blackbody as bb 
import astropy.constants as const
from astropy.io import fits
from scipy.interpolate import interp2d

class FaiglerMazehFit():

	def __init__(self, P_orb, inc, R_star, M_star, T_star, A_ellip=False, A_beam=False,
				 R_p=False, a=False, u=False, g=0.65, logg=None, tele='TESS', M_p=False,
				 K=False):
		self.P_orb = P_orb 	   		 # orbital period in days
		self.inc = inc * np.pi / 180 # inclination converted to radians
		self.R_star = R_star  		 # radius of the star in solar units
		self.M_star = M_star  		 # mass of the star in solar units
		self.T_star = T_star		 # temperature of the star [K]
		self.A_ellip = A_ellip 		 # ellipsoidal amplitude in ppm
		self.A_beam = A_beam 		 # beaming amplitude in ppm

		self.g = g			  		 # gravity-darkening coefficient, expected range is 0.3-1.0 
		self.logg = logg 			 # log surface gravity of the star [cm s^-2]
		self.tele = tele.lower() 	 # observation instrument used, default is TESS. Only other
									 # other option (for now) is Kepler.
		self.R_p = R_p 				 # radius of the planet in jupiter radii
		self.a = a
		self.u = u 					 # the limb-darkening coefficient, range is 0-1
		self.g = g
		self.M_p = M_p
		self.K = K 
	
		# get the mass from the ellipsoidal amplitude, if given.
		# u is the limb-darkening coefficient, range is 0-1
		if not M_p and not not A_ellip and not not logg:
			self.u = self.LDC()
			self.M_p = self.m_from_ellip()

		# star-planet separation [au] assuming a circular orbit
		if not a and not not M_p:
			self.a = get_a(self.P_orb * 86400, self.M_star * const.M_sun.value, \
						   self.M_p * const.M_jup.value) / const.au.value


	def alpha_ellip(self):
		if not self.u:
			self.u = self.LDC()
		if not self.g:
			self.g = self.GDC()

		a = 15 + self.u
		b = 1 + self.g
		c = 3 - self.u
		return 0.15 * a * b / c

	def RV_amp(self):
		"""
		Returns the radial velocity amplitude [m/s] of the star given a companion mass.
		"""
		return 27 / 40 * const.c.value \
			   * self.M_star ** (-2/3) \
			   * self.P_orb ** (-1/3) \
			   * self.M_p * np.sin(self.inc) 

	def doppler_shift(self, K):
		"""
		Returns the shift in wavelength for a given radial velocity amplitude.
		"""
		return K / const.c.value

	def response_convolution(self, lambdas, response):		
		return response * bb.blackbody_lambda(lambdas, self.T_star).value

	def alpha_beam(self, K):
		""" 
		Returns the factor that accounts for the flux lost when a star gets Doppler shifted
		in and out of the observer's bandpass. 
		"""
		print(K)
		rest_lambdas, response = response_func(self.tele)

		flux_rest = np.trapz(self.response_convolution(rest_lambdas, response), \
							 x=rest_lambdas)

		blueshifted_lambdas = rest_lambdas - self.doppler_shift(K=K)
		flux_blueshift = np.trapz(self.response_convolution(blueshifted_lambdas, response), \
								  x=rest_lambdas)

		redshifted_lambdas = rest_lambdas + self.doppler_shift(K=K)
		flux_redshift = np.trapz(self.response_convolution(redshifted_lambdas, response), \
								 x=rest_lambdas)

		alpha_blue = abs( (flux_rest - flux_blueshift) / flux_rest )
		alpha_red = abs( (flux_rest - flux_redshift) / flux_rest )

		return 1 - np.mean([alpha_red, alpha_blue])


	def m_from_ellip(self):
		return self.A_ellip \
			   * self.R_star ** (-3) \
			   * self.M_star ** 2 \
			   * self.P_orb ** 2 \
			   / (12.8 * self.alpha_ellip() * np.sin(self.inc) ** 2)

	def ellip_from_m(self):
		return self.M_p * 12.8 * self.alpha_ellip() * np.sin(self.inc) ** 2 \
			   * self.R_star ** 3 \
			   * self.M_star ** (-2) \
			   * self.P_orb ** (-2)

	def m_from_beam(self, K=False, alpha_beam=False):
		if not alpha_beam and not K and not not self.M_p:
			alpha_beam = self.alpha_beam(K=self.RV_amp())
		elif not alpha_beam and not not K:
			alpha_beam = self.alpha_beam(K=K)
		elif not not K and not not alpha_beam:
			raise ValueError("Please only specify either K or alpha_beam, not both.")
		elif not K and not alpha_beam:
			raise ValueError("Please specify a radial velocity (K) or alpha_beam parameter")
	
		return self.A_beam \
			   * self.M_star ** (2/3) \
			   * self.P_orb ** (1/3) \
			   / (alpha_beam * np.sin(self.inc) * 2.7)


	def beam_from_m(self):
		"""
		Returns the expected Doppler beaming amplitude [ppm] for a given mass.
		"""
		if not self.M_p:
			raise ValueError("Argument 'M_p' must be specified if you're trying to " +  
							 "derive a beaming amplitude from a mass.")

		if not self.K:
			K=self.RV_amp()
				  
		return 2.7 * self.alpha_beam(K=self.K) \
			   * self.M_star ** (-2/3) \
			   * self.P_orb ** (-1/3) \
			   * self.M_p * np.sin(self.inc)

	def Ag_from_thermref(self, A_thermref):
		"""
		Return the geometric albedo derived from the thermal + ref amplitude.
		"""
		return  A_thermref * (self.R_p / self.a) ** -2 * (const.au / const.R_jup) ** 2

	def mass(self, derived_from=None, K=False, alpha_beam=False):

		if derived_from == "ellip":
			return self.m_from_ellip()

		elif derived_from == "beam":
			return self.m_from_beam(K=K, alpha_beam=alpha_beam)

		else:
			raise ValueError("derived_from must equal either 'ellip' or 'beam'")

	def nearest_neighbors(self, value, array, max_difference):
		""" 
		Returns a set of nearest neighbor indices of the given array.
		"""
		return set(list((np.where(abs(array - value) < max_difference))[0]))

	def correct_maxdiff(self, value, array, guess):
		while len(self.nearest_neighbors(value, array, guess)) > 0:
			guess -= 0.01 * guess
		return guess

	def shared_neighbor(self, value1, array1, max_diff1, value2, array2, max_diff2):
		set1 = self.nearest_neighbors(value1, array1, max_diff1)
		set2 = self.nearest_neighbors(value2, array2, max_diff2)
		nearest = list(set1.intersection(set2))

		# if len(nearest) > 1:
		# newmax_diff1 = self.correct_maxdiff(value1, array1, max_diff1)
		# newmax_diff2 = self.correct_maxdiff(value2, array2, max_diff2)
		# print(newmax_diff1, newmax_diff2)
		# if newmax_diff2 > newmax_diff1:
		# 	max_diff2 = newmax_diff2
		# else:
		# 	max_diff1 = newmax_diff1

		# set1 = self.nearest_neighbors(value1, array1, max_diff1)
		# set2 = self.nearest_neighbors(value2, array2, max_diff2)
		# nearest = list(set1.intersection(set2))
		# print(nearest)

		# # if len(nearest) > 1:

		# # 	raise ValueError("Multiple shared nearest neighbors, indices = ", nearest)
		# # else:
		# # 	return nearest[0]

		return nearest[0]


	def tess_warning(self):
		if self.tele != 'tess':
			raise ValueError("This function is only appropriate for observations done with " +
							 "the TESS satellite")

	def claret_LDC(self):
		"""
		Returns the mu coefficient and the four-parameters used in the Claret four-parameter 
		limb-darkening law (Claret 2000). These are obtained by finding the nearest neighbor
		in the model limb-darkening of TESS from Claret 2018. 
		"""
		# print("claret_LDC is still garbage, sorry. Quitting now...")
		# exit()
		self.tess_warning()

		logg, Teff, a1, a2, a3, a4, mu, mod = np.genfromtxt('../claret_ldc.dat', 
														usecols=(0,1,4,5,6,7,8,10),
														unpack=True)
		mod = np.genfromtxt('../claret_ldc.dat', usecols=(10,), dtype='str')

		if self.T_star <= 3000:
			# the PC model is meant for cool stars, and if we break it up this way we can do an 
			# easier 2D interpolation.
			mask = mod == 'PD'
		else:
			mask = mod == 'PC'
		logg = logg[mask]
		Teff = Teff[mask]
		a1 = a1[mask]
		a2 = a2[mask]
		a3 = a3[mask]
		a4 = a4[mask]
		mu = mu[mask]

		nearest = self.shared_neighbor(self.T_star, Teff, 100, self.logg, logg, 0.25)

		mu = mu[nearest]
		a_coeffs = [a1[nearest], a2[nearest], a3[nearest], a4[nearest]]

		return mu, a_coeffs

	def GDC(self):
		"""
		Returns the gravity-darkening coefficient from the Claret 2017 model
		"""
		self.tess_warning()

		logg, log_Teff, g = np.genfromtxt('../claret_gdc.dat', usecols=(2,3,4), unpack=True)
		
		nearest = self.shared_neighbor(np.log10(self.T_star), log_Teff, .01, self.logg, 
					   				   logg, 0.25)
		return g[nearest]

	def LDC(self):
		"""
		Returns the limb-darkening coefficient of the host star.
		"""
		mu, a_coeffs = self.claret_LDC()
		return 1 - sum([a_coeffs[k] * (1 - mu ** ((k+1) / 2)) for k in range(4)])

def get_response_specs(tele):
	if tele=="tess":
		return "../tess-response-function-v1.0.csv", ',', 1e1
	elif tele=="kepler":
		return "../kepler_hires.dat", '\t', 1e4

def response_func(tele):
	file, delimiter, to_AA = get_response_specs(tele)
	lambdas, response = np.genfromtxt(file, delimiter=delimiter, usecols=(0,1), unpack=True)
	return lambdas * to_AA, response

def get_a(P, M_star, M_p):
	"""
	Use Kepler's third law to derive the star-planet separation.
	"""
	return (P ** 2 * const.G.value * (M_star + M_p) / (4 * np.pi ** 2)) ** (1/3)

#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import sys, csv, scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, pacf
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': '14.0'}
plt.rc('font', **font)


def autocorrelation_plot(series, ax=None, **kwds):
    """Autocorrelation plot for time series.

    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method

    Returns:
    -----------
    ax: Matplotlib axis object
    """
#    import matplotlib.pyplot as plt
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = map(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.axhline(y=z95 / np.sqrt(n), color='grey')
    ax.axhline(y=0.0, color='black')
    ax.axhline(y=-z95 / np.sqrt(n), color='grey')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    #ax.set_title("Timeseries ACF")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax

def AC_test(x):
	''' Autocorrelation criterion test for trend presence in data.
	'''
	n = x.size;	r = 0; r_numer = 0; r_denom = 0; r_norm = 0; ac_sum = 0
	for i in xrange(n-1):
		ac_sum += x[i] * x[i+1]
	r_numer = n * ac_sum - pow(np.sum(x), 2) + n * x[0] * x[n-1]
	r_denom = n * np.sum(map(lambda x: pow(x, 2), x)) - pow(np.sum(x), 2)
	r = r_numer / r_denom
	E_norm = (-1) / (n - 1); D_norm = n * (n - 3) / (pow(n-1, 2) * (n + 1))
	r_norm = (r - E_norm) / np.sqrt(D_norm)
	return r_norm

def adftest(y, short_flag):
	'''Augmented Dicky-Fuller test for given timeseries.
	When test-statistics (first returned value) is absolutely less than critical values,
	process could be considered as stationary one.'''
	sep = 32 * '--'
	print "\n\t\tAugmented Dicky-Fuller test\n"
	if short_flag:
		stationarity = ["stationary", "nonstationary"]

		test_c = adfuller(y, regression='c')
		stat_c = 1 if test_c[0] > test_c[4]['5%'] else 0

		test_ct = adfuller(y, regression='ct')
		stat_ct = 1 if test_ct[0] > test_ct[4]['5%'] else 0

		test_ctt = adfuller(y, regression='ctt')
		stat_ctt = 1 if test_ctt[0] > test_ctt[4]['5%'] else 0
		
		test_nc = adfuller(y, regression='nc')
		stat_nc = 1 if test_nc[0] > test_nc[4]['5%'] else 0

		print sep
		print "- constant only:\t\t\t\t{}".format(stationarity[stat_c])
		print "- constant and trend:\t\t\t\t{}".format(stationarity[stat_ct])
		print "- constant, and linear and quadratic trend:\t{}".format(stationarity[stat_ctt])
		print "\n- no constant, no trend:\t\t\t{}".format(stationarity[stat_nc])
		print sep	
	else:
		print "- constant only\n{}".format(adfuller(y,regression='c'))
		print "- constant and trend\n{}".format(adfuller(y,regression='ct'))
		print "- constant, and linear and quadratic trend\n{}".format(adfuller(y,regression='ctt'))
		print "\n- no constant, no trend\n{}".format(adfuller(y,regression='nc'))
		print sep

def Bartels_test(x):
	''' Bartels test for trend presence in data.
	'''
	# Obtain array containing ranks of corresponding values in x-array
	#ranks = np.empty(x.size, int)
	#ranks[x.argsort()] = np.arange(x.size)
	ranks = x.argsort().argsort()
	rank_av = np.mean(ranks)
	r_numer = 0; r_denom = 0

	for i in xrange(ranks.size-1):
		r_numer += pow(ranks[i] - ranks[i+1], 2)
	for i in xrange(ranks.size):
		r_denom += pow(ranks[i] - rank_av, 2)
	b = r_numer / r_denom
	b_norm = (b - 2) / (2 * np.sqrt(5 / (5 * x.size + 7)))
	return b_norm

def distribution_estimate(data, distributions, verb_level=3, f_plot_pdf=True):
	''' Estimates best fit parameters and likelihood of given data
	for each distribution from list. 
	'''
	# Arrays to store results
	parameters = []; llvalue = []
	# Verify distributions
	for dist in distributions:
		# Choose distribution family
		d = getattr(scipy.stats, dist)
		# Fit parameters
		params = d.fit(data)
		parameters.append(params)
		# Estimate likelihood
		llvalue.append(LL_estimate(data, d, *params))
	assert(len(llvalue) == len(parameters))
	
	# Print results
	ranged_indexes = np.argsort(llvalue)
	sep = 20 * "----"
	print "{}\n  Distribution\tLikelihood\t\t\tParameters\n{}".format(sep, sep)
	for i in xrange(1, verb_level+1):
		print "%14s: %10.4f   %s" % (dist_names[ranged_indexes[-i]], llvalue[ranged_indexes[-i]], parameters[ranged_indexes[-i]])
	print sep

	# Plot results
	if f_plot_pdf:
		fig1 = plt.figure()
		x = np.linspace(min(data), max(data), data.size)
		h = plt.hist(data, bins=np.linspace(min(data), max(data), 20), normed=True)
		for i in xrange(len(dist_names)):
			d = getattr(scipy.stats, distributions[i])
			plt.plot(x, d.pdf(x, *parameters[i][:-2], loc=parameters[i][-2], scale=parameters[i][-1]), label=dist_names[i])
		plt.xlim(min(data), max(data))
		plt.ylim(0, 1.5*max(h[0]))
		plt.xlabel(u'Значение')
		plt.ylabel(u'Вероятность появления')
		plt.legend(loc='best')
		plt.grid(True)

def exclude_trend(ts, trend_order=0, excl_trend_plot=False):
	'''Remove trends of defined order from time series.
	'''
	ts_length = len(ts)
	# Obtain trend parameters
	trend = np.polyfit(np.arange(ts_length), ts, trend_order)	
	# Remove trend
	ts_excl_trend = ts
	for t_ord in xrange(trend_order+1):
		ts_excl_trend = ts_excl_trend - pow(np.arange(ts_length), t_ord) * trend[trend_order - t_ord]
	if excl_trend_plot:
		fig4 = plt.figure()
		plt.plot(np.arange(ts_length), ts, color='k', marker='.', label=u'Исходный ряд')
		plt.plot(np.arange(ts_length), ts_excl_trend, color='k', marker='x', label=u'Удален тренд порядка {}'.format(trend_order))
		plt.legend(loc='best')
		plt.grid(True)
	return ts_excl_trend

def hurst_RS(ts, maxlag='auto', f_plot=True):
	'''Returns the Hurst exponent of the time series by calculating R/S-statistic.
	'''
	ts_length = len(ts)
	# Number of lags to investigate
	if maxlag == 'auto':
		# Use only 30% lowest lags
		maxlag = int(round(0.3*ts_length/2))
		if maxlag < 5:
			maxlag = 5
	elif maxlag == 'all':
		maxlag = int(ts_length / 2)
	lags = xrange(2, maxlag+1)

	# Calculate R/S-statistics on different lags
	RS = []
	for lag in lags:
		R = []; S = []
		for i in xrange(int(ts_length / lag)):
			# Calculate centred timeseries in block
			ts_centred = ts[i*lag:(i+1)*lag] - np.repeat(np.mean(ts[i*lag:(i+1)*lag]),lag)
			# Maximum range in block with centred values
			R.append(max(np.cumsum(ts_centred)) - min(np.cumsum(ts_centred)))
			# Standart deviation in block
			S.append(np.std(ts[i*lag:(i+1)*lag]))
		# Residuals of time series divided on blocks (block_size = current lag)
		if ts_length % lag != 0:
			ts_centred = ts[-(ts_length%lag)-1:-1] - np.repeat(np.mean(ts[-(ts_length%lag)-1:-1]),ts_length%lag)
			R.append(max(np.cumsum(ts_centred)) - min(np.cumsum(ts_centred)))
			S.append(np.std(ts[i*lag:(i+1)*lag]))
		# Mean normed range for current lag size
		RS.append(np.mean(np.asarray(R)/np.asarray(S)))

	# Obtain approximation line slope and get Hurst exponent
	poly = np.polyfit(np.log(lags), np.log(RS), 1)
	hurst = poly[0]

	# Plot results
	if f_plot:
		# R/S-statistics on full range of possible lags
		RS_full = []
		for lag in xrange(2, int(ts_length / 2)):
			R_full = []; S_full = []
			for i in xrange(int(ts_length / lag)):
				ts_centred = ts[i*lag:(i+1)*lag] - np.repeat(np.mean(ts[i*lag:(i+1)*lag]),lag)
				R_full.append(max(np.cumsum(ts_centred)) - min(np.cumsum(ts_centred)))
				S_full.append(np.std(ts[i*lag:(i+1)*lag]))
			if ts_length % lag != 0:
				ts_centred = ts[-(ts_length%lag)-1:-1] - np.repeat(np.mean(ts[-(ts_length%lag)-1:-1]),ts_length%lag)
				R_full.append(max(np.cumsum(ts_centred)) - min(np.cumsum(ts_centred)))
				S_full.append(np.std(ts[i*lag:(i+1)*lag]))
			# Mean normed range for current lag size
			RS_full.append(np.mean(np.asarray(R_full)/np.asarray(S_full)))
		# New figure
		#fig3 = plt.figure()
		plt.subplot(122)
		x = np.log(xrange(2, int(ts_length / 2)))
		# Plot values
		plt.plot(x, np.log(RS_full), color='r', marker='o', label=u'Реальные значения')
		plt.plot(np.log(lags), poly[0]*np.log(lags)+poly[1], color='g', linewidth=3, label=u'Аппроксимация')
		# Plot borders
		plt.plot(x, np.repeat(poly[1], x.size), linestyle='--', color='b')
		plt.plot(x, 0.5*x+poly[1], linestyle='--', color='b')
		plt.plot(x, x+poly[1], linestyle='--', color='b')
		plt.axvline(np.log(lags[0]), linestyle='--', color='b')
		plt.axvline(np.log(lags[-1]), linestyle='--', color='b')
		plt.xlabel(u'Log(lag)')
		plt.ylabel('Log(R/S)')
		plt.title(u'R/S-статистика')
		plt.legend(loc='best')
	return hurst, maxlag

def hurst_var(ts, maxlag='auto', f_plot=True):
	'''Returns the Hurst exponent of the time series by variance-plot method.
	'''
	ts_length = len(ts)
	# Number of lags to investigate
	if maxlag == 'auto':
		# Use only 30% lowest lags
		maxlag = int(round(0.3*ts_length/2))
		if maxlag < 5:
			maxlag = 5
	elif maxlag == 'all':
		maxlag = int(ts_length / 2)
	lags = xrange(2, maxlag+1)
	# Calculate variances on different lags
	var_array = []
	for lag in lags:
		av_series = []
		for i in xrange(int(ts_length / lag)):
			av_series.append(np.mean(ts[i*lag:(i+1)*lag]))
		if ts_length % lag != 0:
			av_series.append(np.mean(ts[-(ts_length%lag)-1:-1]))
		var_array.append(np.var(av_series))
	# Obtain approximation line slope and get Hurst exponent
	poly = np.polyfit(np.log(lags), np.log(var_array), 1)
	hurst = 1 + poly[0]/2
	# Plot results
	if f_plot:
		# Variances on full range of possible lags
		var_full = []
		for lag in xrange(2, int(ts_length / 2)):
			av_series = []
			for i in xrange(int(ts_length / lag)):
				av_series.append(np.mean(ts[i*lag:(i+1)*lag]))
			if ts_length % lag != 0:
				av_series.append(np.mean(ts[-(ts_length%lag)-1:-1]))
			var_full.append(np.var(av_series))
		# New figure
		fig3 = plt.figure()
		plt.subplot(121)
		x = np.log(xrange(2, int(ts_length / 2)))
		# Plot values
		plt.plot(x, np.log(var_full), color='r', marker='o', label=u'Реальные значения')
		plt.plot(np.log(lags), poly[0]*np.log(lags)+poly[1], color='g', linewidth=3, label=u'Аппроксимация')
		# Plot borders
		plt.plot(x, np.repeat(poly[1], x.size), linestyle='--', color='b')
		plt.plot(x, -1*x+poly[1], linestyle='--', color='b')
		plt.plot(x, -2*x+poly[1], linestyle='--', color='b')
		plt.axvline(np.log(lags[0]), linestyle='--', color='b')
		plt.axvline(np.log(lags[-1]), linestyle='--', color='b')
		plt.xlabel(u'Log(lag)')
		plt.ylabel('Log(Variance)')
		plt.title(u'График изменения дисперсии')
		plt.legend(loc='best')
	return hurst, maxlag

def hypoteze_check(statistics, quantile='95%'):
	norm_quantiles = {'99.99%': 3.715, '99.9%': 3.090, '99%':	2.326, '97.72%': 2.000, '97.5%': 1.960, '95%': 1.645, '90%': 1.282, '84.13%': 1.000, '50%': 0.000}
	if abs(statistics) < norm_quantiles[quantile]:
		H0 = True
	else:
		H0 = False
	return H0

def LL_estimate(data, distribution, *params):
	''' Estimates log-likelihood value of defined probability law.
	'''
	return sum(np.log(distribution.pdf(data, *params)))

def loadData(filename):
	'''Loads timeseries from file'''
	x = []; y = []
	i = 0
	fd = open(filename, 'rU')
	c = csv.reader(fd)
	for row in c:
		i += 1
		if len(row) != 0:
			try:
				x.append(float(row[0]))
				y.append(float(row[1]))
			except ValueError:
				print "Inappropriate data detected in row {}: {}!".format(i, row)
		else:
			print "String {} is empty!".format(i)
	fd.close()
	return np.asarray(x), np.asarray(y)

def trend_test(data):
	ac = hypoteze_check(AC_test(data))
	bart = hypoteze_check(Bartels_test(data))
	print "\n\n  Test for trend presence in data\n"
	sep = 20 * '--'
	print "{}\n    Test\t\t  Result\n{}".format(sep, sep)
	print "Autocorrelation\t\t{}".format("No trend" if ac else "Trend detected")
	print "Bartels \t\t{}".format("No trend" if bart else "Trend detected")
	print "{}\n".format(sep)


if __name__ == "__main__":

	# Control flags
######################################################
	# Maximum lag step for Hurst log-log estimation 
	# (number of steps or 'auto' for 30% lowest or 'all' for whole timeseries)
	maxstep = 'auto'
	# Plot Hurst exponent?
	f_plot_Hurst = False
	# If flag is set in true, PACF will be shown, otherwise - power spectrum
	f_show_PACF = True
	# Number of lags shown on PACF plot (if None, than all timeseries used)
	pacf_lags = 40
	# ACF type flag
	f_use_symmetric_ACF = False
	# Plot pdf-functions in distribution estimation test
	f_plot_pdf = False
	# Order of trend to exclude before spectral, ACF and PACF analysis (0 for no trend)
	exclude_trend_order = 3
	# Plot original timeseries vs. one with excluded trend
	f_excl_plot = False
#######################################################

	# Check input arguments
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print "\nUsage: ./timeseries_analyze <file_name> [-s]\n"
		exit(1)

	# Load data from file
	x, y = loadData(sys.argv[1])
	timestep = (x[-1] - x[0]) / (x.size - 1)
	print "Timeseries length: {} points".format(y.size)
	print "Timestep: {} sec".format(timestep)

	# List of used distributions to verify
	dist_names = ['alpha', 'beta', 'cauchy', 'expon', 'gamma', 'lognorm', 'norm', 'pareto', 'powerlaw', 'rayleigh', 'uniform', 'weibull_min', 'weibull_max']
	# Obtain data probability distribution law by MLE
	print "\n\t\tData distribution test\n"
	distribution_estimate(y, dist_names, f_plot_pdf=f_plot_pdf)

	# Trend presence test
	trend_test(y)

	# ADF-test for stationarity
	# Flag to shorten ADF-test output
	is_adf_short = True
	adftest(y, is_adf_short)

	# Exclude trend for further analysis
	y = exclude_trend(y, trend_order=exclude_trend_order, excl_trend_plot=f_excl_plot)

	# Estimate Hurst coefficient for timeseries
	h1, ml1 = hurst_var(y, maxstep, f_plot_Hurst)
	h2, ml2 = hurst_RS(y, maxstep, f_plot_Hurst)
	print "\nHurst parameter estimation, maximum_lag: {}".format(ml1)
	print "Variance plot: {}\nR/S-statistic: {}\n".format(round(h1,3), round(h2,3))
	
	# Caclulate timeseries spectrums
	spectrum = abs(np.fft.fft(y))
	freq = abs(np.fft.fftfreq(x.size, timestep))
	power_spectrum = spectrum ** 2

	# Plot results
	fig2 = plt.figure()
	# Plot original timeseries
	plt.subplot(221)
	plt.plot(x, y, color='k')
	plt.xlabel(u"Время")
	plt.ylabel(u"Значение")
	#plt.title("Original timeseries")
	plt.grid(True)

	# ACF plot
	if f_use_symmetric_ACF:
	# Symmetric ACF
		plt.subplot(222)
		plt.acorr(y, maxlags=None, color='k')
		plt.xlabel(u"Время")
		plt.ylabel(u"АКФ")
		#plt.title("Timeseries ACF")
		plt.grid(True)
	else:
	# Asymmetric ACF
		autocorrelation_plot(y, ax=plt.subplot(222), color='k')
		plt.xlabel(u'Шаг')
		plt.ylabel(u'АКФ')
		plt.title('')
	
	# Spectrum plot
	plt.subplot(223)
	plt.plot(freq, np.log(spectrum), color='k')
	plt.xlabel(u"Частота")
	plt.ylabel(u"Амплитуда (log)")
	#plt.title("Timeseries spectrum")
	plt.grid(True)

	if f_show_PACF:
		from statsmodels.graphics.tsaplots import plot_pacf
		plot_pacf(y, ax=plt.subplot(224), lags=pacf_lags)
		plt.xlabel(u'Шаг')
		plt.ylabel(u'ЧАКФ')
		plt.title('')

	else:
		# Power spectrum plot
		plt.subplot(224)
		plt.plot(freq, power_spectrum, color='k')
		plt.xlabel(u"Частота")
		plt.ylabel(u"Мощность")
		#plt.title("Power spectrum")
	plt.grid(True)

	plt.subplots_adjust(hspace=0.4, wspace=0.4)

	# Save results
	if len(sys.argv) == 3 and sys.argv[2] == "-s":
		fname = sys.argv[1].split("/")[-1] + "-analyze.png"
		plt.savefig(fname, dpi=300)
		print "Plots are saved in file: {}\n".format(fname)
	plt.show()
	


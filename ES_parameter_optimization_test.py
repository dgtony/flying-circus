#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': '14.0'}
plt.rc('font', **font)

def loadData(filename):
	'''Load timeseries from file'''
	x = []; y = []
	i = 0
	fd = open(filename, 'rb')
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


def SSE(alpha, data):
	'''Sum of squared errors function. Errors are differences between predicted and real values
	in some timeseries, while forecasting it with exponential smoothing'''
	# Tested timeseries length
	length = data.size

	# Initialise variables
	sse = 0; predict = np.zeros(length)
	predict[:2] = data[:2]
	for i in xrange(1, length - 1):
		# Prediction
		predict[i+1] = exp_smoothing(data[i], alpha, predict[i])
		# Current step squared error estimation
		sse += pow((predict[i+1] - data[i+1]), 2)

	# Return mean sum of squared errors
	return (sse / (length - 2))


def alpha_optimize(data):
	'''Performes smoothing parameter <alpha> optimization for defined sample timeseries.
	'''
	# Initial alpha-value
	alpha0 = 0.2

	# Find optimal alpha value (nelder-mead or BFGS). For debug add option <'disp': 'False'>
	result = minimize(SSE, alpha0, args=(data,), method='nelder-mead', options={'maxiter': '100'})
	alpha = result.x
	
	return alpha


def exp_smoothing(x, a, s):
	'''Calculates simple exponential smoothing:
	S(t) = a * x(t) + (1 - a) * S(t - 1)
	Value S(t) in such model is a forecast for point (t + 1)'''
	return a * x + (1 - a) * s


def test_alpha_optimization(data, length):
	'''Consecutive training series increase and further alpha and SSE estimation.
	'''
	alpha = np.zeros(length - 2); sse = np.zeros(length - 2)
	assert(length <= data.size)
	for i in xrange(length - 2):
		# Progress indicator
		sys.stdout.write("\r%d%%" % round(100 * i / (length - 2)))
		sys.stdout.flush()

		# Alpha optimization procedure
		alpha[i] = alpha_optimize(data[:i + 3])

		# Compute SSE for full timeseries with current <alpha> value
		sse[i] = SSE(alpha[i], data)

	# Return obtained arrays with optimal alpha values and corresponding mean squared errors sums
	return alpha, sse


if __name__ == "__main__":
	# Check arguments
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print "Utility usage: ./{} <file_name> [-s]".format(sys.argv[0])
		exit(0)

	# Load data
	print "Loading data..."
	x, y = loadData(sys.argv[1])

	# Use factor to avoid overflow
	factor = 1e-6
	y = y * factor

	# Get full array length
	data_length = y.size
	if data_length < 3:
		print "Timeseries is too short!"
		exit(0)
	assert(x.size == data_length)

	# Run test
	print "\nProcessing..."
	alpha, sse = test_alpha_optimization(y, data_length)

	# Plot obtained results
	figure1 = plt.figure()
	plt.plot(xrange(data_length - 2), alpha)
	plt.axhline(y=1, linewidth=1, linestyle='--')
	plt.axhline(y=0, linewidth=1, linestyle='--')
	plt.xlabel(u'Длина интервала подгонки сглаживающего коэффициента')
	plt.ylabel(u'Значение сглаживающего коэффициента')
	plt.title(u'Выбор сглаживающего коэффициента')
	plt.grid(True)
	# Save plot
	if len(sys.argv) == 3 and sys.argv[2] == '-s':
		fname = sys.argv[1].split("/")[-1] + "-alpha.png"
		plt.savefig(fname, dpi=300)

	figure2 = plt.figure()
	plt.plot(xrange(data_length - 2), sse)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(u'Длина интервала подгонки сглаживающего коэффициента')
	plt.ylabel(u'Средняя квадратичная ошибка предсказания')
	plt.title(u'Ошибка предсказания')
	plt.grid(True)
	# Save plot
	if len(sys.argv) == 3 and sys.argv[2] == '-s':
		fname = sys.argv[1].split("/")[-1] + "-SSE.png"
		plt.savefig(fname, dpi=300)
		print "\nPlots are saved in corresponding files."

	plt.show()





# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import spline
import sys
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': '14.0'}
plt.rc('font', **font)

'''Utility used to plot and animate predictor-filter spectrums, obtained during prediction procedure. 
Use only input files saved from LP-test performed by "predict"-utility!
'''
# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(freq, trans_fun[i])
    return line,

def loadData(filename):
	'''Load coefficients from file to list of numpy arrays'''
	lst = [];
	fd = open(filename, 'rb')
	for string in fd:
		coeff = [float(s) for s in string.strip().split(" ")]
		lst.append(np.asarray(coeff, dtype=float))
	fd.close()
	return lst


if __name__ == "__main__":
	# Check arguments
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print "Usage: ./{} <file> [-a]".format(sys.argv[0])
		print "To save animation in video file use key \"-a\""
		exit(0)

	# Load data
	coeff = loadData(sys.argv[1])

	# Get signal length from filename
	sig_length = int((sys.argv[1].partition("sig")[2]).split(".")[0])

	# Compute filter spectrums
	trans_fun = []
	for filt in coeff:
		filter_realisation = np.concatenate((filt, np.zeros(sig_length - len(filt) + 1)))
		trans_fun.append(1./abs(np.fft.rfft(filter_realisation)))
	freq = np.fft.rfftfreq(sig_length, 1)

	if len(sys.argv) == 3 and sys.argv[2] == "-a":
		# Plot results
		fig = plt.figure()
		ax = plt.axes(xlim=(min(freq), max(freq)), ylim=(0, max([max(v) for v in trans_fun])))
		line, = ax.plot([], [], lw=2)
		# Animate
		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(trans_fun), interval=100, blit=True)
		anim.save(sys.argv[1] + '.mp4', fps=24)
		
		
	# Make plot
	plt.clf()
	# Smooth lines with interpolation
	xnew = np.linspace(min(freq), max(freq), 100); y = []

	for i in xrange(len(trans_fun)):
		ynew = spline(freq, trans_fun[i], xnew, order=3)
		plt.plot(xnew, ynew)
	plt.xlabel(u"Частота")
	plt.ylabel(u"Амплитуда")
	plt.title(u"АЧХ фильтра-предсказателя")
	plt.grid(True)
	#plt.savefig(sys.argv[1]+".png", dpi=300)
	plt.show()





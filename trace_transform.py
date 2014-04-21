#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import csv,sys
import matplotlib.pyplot as plt
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': '14.0'}
plt.rc('font', **font)

# Color to plot
color = ('g','b','r','#6F4E37','m','c','y','k','*','o','--')
# Figure resolution (for saving)
res = 300

'''Utility to transform original csv-files containing traffic load traces into averaged ones.
Averaging is performed by time intervals defined in seconds, original intervals between neighbour 
values are obtained from first column in data file (in assumption, that this interval is constant).
Original data file should have only two columns with data, separated by commas:
	<current_time>, <current_load>
Other formats will cause errors. Results will be saved in single files'''

# Save results in file
def save(fname, step, x, y):
	splitter = ","
	outfile = open(fname,'wb')
	title = "Trace averaged with timestep {}".format(step)
	outfile.write(title + "\n")
	for r in xrange(len(x)):
		string=str(x[r]) + splitter + str(y[r])
		outfile.write(string + "\n")	
	outfile.close()

# Main
if len(sys.argv) < 3:
	print "Utility usage:\n\t./trace_transform <filename.csv> <list_of_steps> [-s]"
	print "Option -s used to save results in corresponding files.\n"
	sys.exit(0)

xl = []; yl = []; i = 0
fname = sys.argv[1].split(".")
fname_part = ".".join(fname[:-1])
print "Open file \"{}\", processing...".format(sys.argv[1])
fd = open(sys.argv[1], 'rb')
c = csv.reader(fd)
for row in c:
	i += 1
	if len(row) != 0:
		try:
			xl.append(float(row[0]))
			yl.append(float(row[1]))
		except ValueError:
			print "Inappropriate data detected in row {}: {}!".format(i, row)
	else:
		print "String {} is empty!".format(i)

# Check obtained arrays for length equality
if (len(xl) != len(yl)):
	print "Wrong input data! X and Y arrays must be of same length"
	sys.exit(1)

# Get steps from argument
step = []
llist = sys.argv[2].split(",")
for i in xrange(len(llist)):
	step.append(int(llist[i]))

# Get original parameters
x_start = int(xl[0])
x_stop = int(xl[-1])
x_step = (x_start - x_stop) / (len(xl) - 1)

# Averaging by defined steps
x = []; y = []
for i in xrange(len(step)):
	if step[i] < x_step:
		print "Error! Step %2.2f is less than original!\n" % (step[i])
		exit(1)
	x.append([])
	x[i] = range(x_start, x_stop, step[i])
	
	# Y-array
	y.append([])
	for p in x[i]:
		s = 0; n = 0
		for k in xrange(len(xl)):
			if xl[k] >= p and xl[k] < (p + step[i]):
				s += yl[k];	n += 1
		y[i].append(s / n)
	x[i] = [(val + step[i] / 2) for val in x[i]]

# Plot results
plt.plot(xl, yl, color[0], label=u"Исходный")
for i in xrange(len(step)):
	plt.plot(x[i], y[i], color[i + 1], linewidth=2.0, label=u"Шаг {}".format(step[i]))
plt.xlabel(u'Время, сек')
plt.ylabel(u'Интенсивность трафика, байт/сек')
plt.title(u"Сглаживание временного ряда") 
plt.legend(loc='best')
plt.grid(True)

# Save results in files
if (len(sys.argv) == 4 and sys.argv[3] == '-s'):
	plt.savefig(fname_part + "-av.png", format='png', dpi=res)
	for i in xrange(len(step)):
		outfile = fname_part + "-av{}.csv".format(step[i])
		save(outfile, step[i], x[i], y[i])
	print "Averaged timeseries are saved in corresponding files."

plt.show()

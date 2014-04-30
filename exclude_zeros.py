#!/usr/bin/python
import numpy as np
import sys, csv

''' Exclude all zero y-values from given file and write it out with new x-scale.
'''
# Check argument
if len(sys.argv) < 2:
	print "Utility usage:\n\t{} <filename.csv>".format(sys.argv[0])
	sys.exit(0)

# Load data from file
xl = []; yl = []; i = 0
fd = open(sys.argv[1],'rb')
c = csv.reader(fd)
for row in c:
	i += 1
	if len(row) != 0:
		try:
			xl.append(float(row[0]))
			yl.append(float(row[1]))
		except ValueError:
			print "Inappropriate data detected in row {}: {}!".format(i,row)
	else:
		print "String {} is empty!".format(i)
x = np.asarray(xl)
y = np.asarray(yl)
assert(x.size==y.size)

# Exclude null y-values
y_new = []
for pointer in xrange(y.size):
	if y[pointer] != 0:
		y_new.append(y[pointer])
	else:
		continue
y_new = np.asarray(y_new)

# New x-array
x_new = np.arange(y_new.size)
assert(x_new.size==y_new.size)

# Write to file
splitter = ','
outfname = '.'.join(sys.argv[1].split('.')[:-1]) + '-eg.csv'
outfd = open(outfname,'wb')
for spointer in xrange(x_new.size):
	string = str(x_new[spointer]) + splitter + str(y_new[spointer])
	outfd.write(string + "\n")	
outfd.close()
print "Result saved in file: ", outfname

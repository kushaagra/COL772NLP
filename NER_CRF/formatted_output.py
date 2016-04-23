import re
import sys
input_file = sys.argv[1]
output_file = sys.argv[2]
f1 = open(input_file,'r')
f2 = open(output_file,'w')
f3 = open('my_out','r')
lines = f1.readlines()
results = f3.readlines()
for i in range(len(lines)):
	line1 = lines[i]
	line1 = line1.rstrip('\n')
	line1 = line1.rstrip('\r')
	f2.write(line1 + " " + results[i])

f1.close()
f2.close()
f3.close()
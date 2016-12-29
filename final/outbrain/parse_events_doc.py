import pandas as pd

output = open('events_document.csv', 'wb')
with open('events.csv') as f:
	for line in f:
		ls = line.split(',')
		ls_sel = [ls[0], ls[2]]
		output.write(",".join(map(str,ls_sel)))
		output.write("\n")

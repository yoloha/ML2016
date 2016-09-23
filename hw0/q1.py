import sys
f = open(sys.argv[2],'r')
temp = f.read().splitlines()
result=[]
for x in temp:
	result.append(float(x.split(' ')[int(sys.argv[1])]))
f.close()
result.sort()
string =  ", ".join(map(str,result))
output = open('ans1.txt','w')
output.write(string)
output.close()

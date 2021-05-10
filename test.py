searchfile = open("words.txt", "r")
for line in searchfile:
    if 'n02190166' in line: 
    	things = line
    	print(things)
thing = things[10:]
print(thing)
searchfile.close()
import numpy as np
import matplotlib.pyplot as plt
import csv

data = []
with open("data/026157_2022-02-16_13-55-49.csv") as file:
	file_reader = csv.reader(file)
	l = 0
	for row in file_reader:
		if len(row) < 5:
			continue
		data.append(row)
del data[0]

#Time = data[0][1]
#SV = data[0][2]
#p = data[0][3]
#c = data[2][:]
#p = data[:][3]

c = [float(data[i][2]) for i, val in enumerate(data)]
p = [float(data[i][3]) for i, val in enumerate(data)]
del c[0]
del p[0]



fig, ax = plt.subplots(1)
plt.plot(c, p)
#plt.style.use('ggplot')
plt.style.use('dark_background')
plt.gca().invert_yaxis()
plt.title('Sound Velocity Profile')
plt.xlabel('SV [m/s]')
plt.ylabel('Depth [m]')
plt.grid()
ax.set_xlim([min(c)-2,max(c)+2])
#plt.show()
#plt.style.use('grayscale')

# Python 3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
plt.rcParams.update({'font.size':16})
import os

path = os.getcwd()+'/81358_fisc_post_processing/'
directory = os.path.abspath(os.path.join(path, os.pardir)) +'/plots/'

### calc_beam function, input is operation frequency and steering angle ###
def calc_beam(freq, angle, ew):
	########### SETUP for beam calc #################
	#ew = 10e-3 # Element width [M] ## TX_h = 15e-3, RX_h = 20e-3, RX_b = 1.4e-3
	cc = 1.8e-3 # Center-center distance [M]
	Ne = 1# No of elements
	cw = 1480.0 # SV [m/s]
	Ro = 0.5 # Observation distance [M]
	v = np.linspace(-90,90,1000) # Observation angles (-90-90 degree)
	os = np.ceil(Ne/8)
	cs = int(Ne/2)
	cs = 1
	shading_gradient = [0.45,0.636,1,0.636,0.45]
	shading_gradient = [0,10**(-29/20),1,10**(-29/20),0]
	shading_gradient = [0,0,1,0,0]
	shading_sections = [os,os,cs,os,os]
	sh = np.repeat(shading_gradient,shading_sections)
	#sh[48] = 0
	#Electrical shading of the transducer
	#sh = np.array([0.4,1,1,0.4]) #Shading for FLS, due to number for rings.
	#sh = np.hamming(Ne) #Shading with an ideal hamming window

	###################################
	v = np.where(v==0, 1e-6, v) # replace zeros in angle array in order to avoid divide by 0

	xe = np.linspace(1,Ne,Ne)*cc # x positions for the elements
	xe = xe-np.mean(xe) # elements centered
	jk = 2*np.pi*1j*freq/cw # Wave number
	ps = np.exp(jk*xe*np.sin(np.deg2rad(angle))) #Phase delay for steering angle
	kj = np.pi*freq*ew/cw*np.sin(np.deg2rad(v)) #Core in equation
	de = np.sin(kj)/kj # element directivity (pressure)
	p = []
	for i in range(len(v)): # Calculate resulting beam for each of the observation angles
		dy = Ro*np.cos(np.deg2rad(v[i]))
		dx = Ro*np.sin(np.deg2rad(v[i]))-xe
		dr2 = dx*dx+dy*dy
		dr = np.sqrt(dr2)
		#pe = np.sum(de[i]*sh*ps*np.exp(jk*dr)/dr) # With vshading on array
		pe = np.sum(de[i]*ps*np.exp(jk*dr)/dr) # No shading
		p.append(pe)
	D = 20*np.log10(np.abs(p)/np.max(np.abs(p)))
	return v,D


frequencies = [470e3]#,470e3,510e3,1000e3] # There is a bug in plot function, put in at least 3 frequencies

# For a single angle, set start and stop to zero, but keep more than 0 in angle_step in order to avoid /
start_angle = 0
stop_angle = 0
angle_step = 5

polar = False #Set polar to true in order to plot polarplot, best for plotting steering angles.
show_beam_width = True #Show beam width in normal mode, should only be True when plotting single angle.

angles = np.arange(start_angle,stop_angle+angle_step,angle_step)
nrows = np.ceil(len(frequencies)/2)
ncols = 2

if polar:
	fig, axs = plt.subplots(int(nrows), int(ncols), figsize=(5*ncols,5*nrows), subplot_kw=dict(projection="polar"))
else:
	if nrows == 1:
		fig, axs = plt.subplots(1, 1, figsize=(10,8))
		#testfig, testax = plt.subplots(1, 1, figsize=(10,8))
	else:
		fig, axs = plt.subplots(int(nrows), int(ncols), figsize=(10,8))



if nrows > 1:
	for index, freq in enumerate(frequencies):
		x = index//ncols
		y = index%ncols
		print("x", x, "y", y)
		for a in angles:
			ba,p = calc_beam(freq,a)
			r = np.deg2rad(ba)
			#print("ba", ba, "r", r)
			if polar:
				x1 = -60
				x2 = 60
				#upper_intersect = np.where(p>=-3)
				#left_upper_intersect = upper_intersect[0][0]
				#right_upper_intersect = upper_intersect[0][-1]
				axs[x,y].plot(r,p)#, label=f"{a} degree")
				#axs[x,y].annotate(f" {round(-1*(ba[left_upper_intersect])+ba[right_upper_intersect],2)} deg\nbeam width", xy=(0, p[left_upper_intersect]-2))

				#axs[x,y].set_xlim(np.deg2rad([x1,x2]))
				axs[x,y].set_ylim(-20,0)
				#axs[x,y].set_thetagrids(np.linspace(x1,x2,13))
				axs[x,y].set_theta_zero_location("N")
			else:
				axs[x,y].plot(ba,p, label=f"{a} degree")
				upper_intersect = np.where(p>=-3)
				left_upper_intersect = upper_intersect[0][0]
				right_upper_intersect = upper_intersect[0][-1]
				if show_beam_width:
					axs[x,y].annotate(f" {round(-1*(ba[left_upper_intersect])+ba[right_upper_intersect],2)} deg\nbeamwidth", fontsize=12, xy=(17, p[left_upper_intersect]-2))
					#axs[x,y].annotate(f"{round(ba[right_upper_intersect],2)} deg", xy=(ba[right_upper_intersect], p[right_upper_intersect]+0.65))
					axs[x,y].plot(ba[left_upper_intersect],p[left_upper_intersect],"or")
					axs[x,y].plot(ba[right_upper_intersect],p[right_upper_intersect],"or")
				if ba[left_upper_intersect] >= 0:
					axs[x,y].set_xlim(ba[left_upper_intersect]*5*-1,ba[right_upper_intersect]*5)
				else:
					axs[x,y].set_xlim(ba[left_upper_intersect]*8,ba[right_upper_intersect]*8)
					axs[x,y].set_xlim(-80,80)
				axs[x,y].set_ylim(-40,0)

			axs[x,y].minorticks_on()
			axs[x,y].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.8)
			axs[x,y].grid(b=True, which="minor", color="#999999", linestyle="--", alpha=0.2)
			axs[x,y].set_xlabel("f="+f"{freq*1e-3} Khz")
			#axs[x,y].legend()
else:
	depths = np.linspace(-4,4, 1000) # NEW

	TX_d = -1 # New
	RX_d = -1.3 # New
	mid_d = -1.15

	#p_sum = []
	for a in angles:
		ba,p = calc_beam(frequencies[0],a, 13e-3)
		r = np.deg2rad(ba)
		'''


		ba_TX, p_TX = calc_beam(frequencies[0],a, 15e-3)
		r_TX = np.deg2rad(ba_TX)
		#TX_dp = TX_d - np.tan(r_TX) #NEW
		TX_d = np.tan(r_TX) #NEW

		ba_RX, p_RX = calc_beam(frequencies[0],a, 10e-3)
		r_RX = np.deg2rad(ba_RX)
		#RX_dp = RX_d - np.tan(r_RX) #NEW

		#p_sum = p_TX + p_RX

		#sum_d = mid_d - np.tan(r_RX)

		Range = 0.1

		x_depths = np.tan(np.deg2rad(ba_TX))*Range
		#dist_idx = np.where(x_depths > 0.3)[0][0]
		shifted_p_RX = np.roll(p_RX, -49)

		summed_p = p_TX + shifted_p_RX
		summed_p = summed_p-max(summed_p)

		## Finding beamwidth
		sum_upper_intersect = np.where(summed_p>=-3)
		sum_left_upper_intersect = sum_upper_intersect[0][0]
		sum_right_upper_intersect = sum_upper_intersect[0][-1]


		fig, ax = plt.subplots(1, figsize=(10,7))
		#ax2 = ax.twinx()
		ax.plot(p_TX, ba_TX, label='TX Beam')
		ax.plot(shifted_p_RX, ba_RX, label='RX Beam')
		ax.plot(summed_p, ba_RX, label='Summed Beam')
		ax.set_ylabel('Degress', color='k')
		ax.set_xlabel('dB', color='k')
		ax.set_ylim([-60, 60])
		ax.set_xlim([-60, 3])
		plt.suptitle("Summation of FISC Acoustic Beams")# at range: "+str(Range)+' m')
		ax.legend(loc='upper right')
		ax.grid()
		#print("hei", TX_left_upper_intersect, TX_right_upper_intersect)

		ax.annotate(f" {round(-1*(ba_TX[sum_left_upper_intersect])+ba_TX[sum_right_upper_intersect],2)} deg\nsummed\nbeamwidth", fontsize=12, xy=(-6, summed_p[sum_left_upper_intersect]+12))
		#axs[x,y].annotate(f"{round(ba[right_upper_intersect],2)} deg", xy=(ba[right_upper_intersect], p[right_upper_intersect]+0.65))
		ax.plot(summed_p[sum_left_upper_intersect], ba_TX[sum_left_upper_intersect],"or")


		ax.plot(summed_p[sum_right_upper_intersect], ba_TX[sum_right_upper_intersect],"or")

		#plt.savefig(directory+'AcousticSum_Angles_1.5m.pdf')



		fig2, ax2 = plt.subplots(1, figsize=(10,7))
		ax2.plot(p_TX, x_depths, label='TX Beam, Cartesian')
		ax2.plot(shifted_p_RX, x_depths, label='RX Beam, Cartesian')
		ax2.plot(summed_p, x_depths, label='Summed Beam, Cartesian')
		ax2.set_ylabel('Meters', color='k')
		ax2.set_xlabel('dB', color='k')
		ax2.set_ylim([-2, 2])
		print(x_depths[sum_left_upper_intersect] - x_depths[sum_right_upper_intersect])
		beamWidth = np.rad2deg(np.arctan((x_depths[sum_left_upper_intersect] - x_depths[sum_right_upper_intersect]) / Range))
		#print("Beamwidth:", beamWidth)
		#ax2.annotate(f" {round(-1*(x_depths[sum_left_upper_intersect])+x_depths[sum_right_upper_intersect],2)} deg\nbeamwidth", fontsize=12, xy=(-5,0.5))#, summed_p[sum_left_upper_intersect]+0.4))
		#axs[x,y].annotate(f"{round(ba[right_upper_intersect],2)} deg", xy=(ba[right_upper_intersect], p[right_upper_intersect]+0.65))
		ax2.plot(summed_p[sum_left_upper_intersect], x_depths[sum_left_upper_intersect],"or")
		ax2.plot(summed_p[sum_right_upper_intersect], x_depths[sum_right_upper_intersect],"or")
		ax2.set_xlim([-60, 3])
		plt.suptitle("Cartesian Summation of Acoustic Beam at range: "+str(Range)+' m')
		ax2.legend(loc='upper right')
		ax2.grid()
		#plt.savefig(directory+'AcousticSum_1.5m.pdf')
		#ax2.legend()


		new_angles = np.degrees(np.arctan(x_depths)/Range)

		fig3, ax3 = plt.subplots(1)
		ax3.plot(summed_p, x_depths)
		ax3.set_title('Combined vertical beam at range: '+str(Range)+' m')
		ax3.set_ylabel('Meters')
		ax3.set_xlabel('dB')
		ax3.set_xlim([-40, 3])
		ax3.set_ylim([-5, 5])
		ax3.grid()
		#plt.savefig(directory+'SummedBeam_1.5m.pdf')
		plt.show()
		quit()
		#summed = []
		#for val in x_depths:
		#	pass

		ax.plot(p_TX, x_depths, label='TX Beam, Cartesian')
		ax.plot(p_RX, x_depths, label='RX Beam, Cartesian')
		ax.plot(p_sum, x_depths, label='Summed Beam, Cartesian')
		ax.set_xlim([-40, 0])
		ax.set_ylim([-3, 3])
		ax.legend()


		fig2, ax2 = plt.subplots(1)
		ax2.plot(p_sum, ba, label='Summed Beam, By angle')
		plt.show()

		quit()


		for i, val in enumerate(TX_dp):
			print("TX_DP:",TX_dp[i])
			print("RX_DP:", RX_dp[i])
			print("Sum p:", p_sum[i])


		#for i in range(len(RX_dp)):
		#	RX_testp = 10**(RX_dp[i]/20)
		#	TX_testp = 10**(TX_dp[i]/20)
		#	p_sum.append(RX_testp+TX_testp)
		#for i, val in enumerate(RX_dp):
		#	print("\n\rAngle:", ba[i])
		#	print("RX_DP", RX_dp[i])
		#	print("psum:", p_sum[i])
		#p_sum = 20*np.log10(np.abs(p_sum)/np.max(np.abs(p_sum)))
		#sum_dp = mid_d - np.tan(r_RX) #NEW
		#r_sum = np.deg2rad(sum_dp)
		testax.plot(p_sum, depths+mid_d, label='Fucked up Beam') # NEW

		testax.plot(p_TX, TX_dp, label='TX Beam') # NEW
		testax.plot(p_RX, RX_dp, label='RX Beam') # NEW
		#testax.set_ylim(depths[-1],depths[0]) # NEW
		testax.set_ylim(0,-3)
		testax.set_xlim(-40,0)
		testax.legend()
		#print("basdasfa", ba, "r", r)
		'''
		if polar:
			x1 = -60
			x2 = 60
			#upper_intersect = np.where(p>=-3)
			#left_upper_intersect = upper_intersect[0][0]
			#right_upper_intersect = upper_intersect[0][-1]
			axs.plot(r,p, label=f"{a} degree")
			#axs[x,y].annotate(f" {round(-1*(ba[left_upper_intersect])+ba[right_upper_intersect],2)} deg\nbeam width", xy=(0, p[left_upper_intersect]-2))

			#axs[x,y].set_xlim(np.deg2rad([x1,x2]))
			axs.set_ylim(-20,0)
			#axs[x,y].set_thetagrids(np.linspace(x1,x2,13))
			axs.set_theta_zero_location("N")
		else:
			axs.plot(ba,p, label=f"{a} degree")
			upper_intersect = np.where(p>=-3)
			left_upper_intersect = upper_intersect[0][0]
			right_upper_intersect = upper_intersect[0][-1]
			if show_beam_width:
				axs.annotate(f" {round(-1*(ba[left_upper_intersect])+ba[right_upper_intersect],2)} deg\nbeamwidth", fontsize=18, xy=(17, p[left_upper_intersect]-2))
				#axs[x,y].annotate(f"{round(ba[right_upper_intersect],2)} deg", xy=(ba[right_upper_intersect], p[right_upper_intersect]+0.65))
				axs.plot(ba[left_upper_intersect],p[left_upper_intersect],"or")
				axs.plot(ba[right_upper_intersect],p[right_upper_intersect],"or")
			if ba[left_upper_intersect] >= 0:
				axs.set_xlim(ba[left_upper_intersect]*5*-1,ba[right_upper_intersect]*5)
			else:
				axs.set_xlim(ba[left_upper_intersect]*8,ba[right_upper_intersect]*8)
				axs.set_xlim(-80,80)
			axs.set_ylim(-40,0)

		axs.minorticks_on()
		axs.grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.8)
		axs.grid(b=True, which="minor", color="#999999", linestyle="--", alpha=0.2)
		axs.set_xlabel("Angle [deg]", fontsize=18)
		axs.set_ylabel("dB", fontsize=18)
		#axs[x,y].legend()
fig.suptitle("Vertical Beam Pattern Simulation of\nTX (13 mm height). f="+f"{frequencies[0]*1e-3} kHz", fontsize=25)
plt.tight_layout()
#fig.patch.set_facecolor("#edf3f5")
plt.savefig('TX_VerticalBeams.pdf')
plt.show()

plt.close()

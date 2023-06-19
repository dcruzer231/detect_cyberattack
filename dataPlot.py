"""
written by Daniel Cruz
This plots the data found in the csv files
both in the time domain and in the frequency domain
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

"""
toy_virt_channels = {
	0 : 'index',
	1 : 'stepper_driver', //power consumption
	2 : 'ps_wall', 		 //power supply AC
	3 : 'driver_ps', 	// driver to the power supply maps channel between power
	4 : 'instchan', 	// pwm modulation from arduino to 
	5 : 'run_gt',		
	6 : 'inst_gt'		//flip-flop on starting on each instruction
}
"""

def fftPlotter(run):
	titles = ["index","stepper_driver", "ps_wall","driver_ps","instchan","run_gt","inst_gt"]
	fig, ax = plt.subplots(len(run)-1,sharex=True)
	
	samplingInterval = 10e3
	samplingFrequency = 1 / samplingInterval;
	
	tpCount     = len(run[0])
	values      = np.arange(int(tpCount/2))
	timePeriod  = tpCount/samplingFrequency
	frequencies = values/timePeriod


	for i,channel in enumerate(run):	
		
		if i == 0:
			continue
		# sp = np.fft.fft(channel)
		# freq = np.fft.fftfreq(len(channel))
		N = len(channel)

		sp = scipy.fftpack.fft(channel)
		freq = np.linspace(0.0,1.0//(2.0*samplingFrequency),N//2)



		ax[i-1].plot(freq,2.0/N * np.abs(sp[:N//2]))
		ax[i-1].set_title(titles[i])

#run should be the 7 channels 
def runPlotter(run):
	titles = ["index","stepper_driver", "ps_wall","driver_ps","instchan","run_gt","inst_gt"]
	fig, ax = plt.subplots(len(run)-1,sharex=True)
	for i,channel in enumerate(run):
		if i == 0:
			continue
		# len(channel)
		# x = np.linspace(0,len(channel))
		ax[i-1].plot(run[0],channel)
		ax[i-1].set_title(titles[i])
	

if __name__ == '__main__':
	plt.close("all")

	sweepPath = "C:\\\\Users\\dcruz\\Documents\\ornl_internship\\Data\\frequency_sweep\\"
	attackPath = "C:\\\\Users\\dcruz\\Documents\\ornl_internship\\Data\\simulated_attack\\known-bad\\"
	goodPath = "C:\\\\Users\\dcruz\\Documents\\ornl_internship\\Data\\simulated_attack\\baseline-known-good\\"


	sweepFiles = ["data-HALF-BLUE-LAB-r10000d1800-091820-182807.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-142849.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-154000.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-170842.h5f"]
	goodFiles = ["data-increment-BL-r10000d18000-051320-125438.h5f","data-increment-BL-r10000d18000-051320-184456.h5f","data-increment-BL-r10000d18000-051420-110816.h5f","data-increment-BL-r10000d18000-051420-171038.h5f"]
	attackFiles = ["data-increment-MUTSPD5-r10000d18000-061520-105843.h5f","data-increment-MUTSPD5-r10000d18000-061620-002749.h5f","data-increment-MUTSPD-r10000d18000-051720-150804.h5f","data-increment-MUTSPD-r10000d18000-051820-114805.h5f","data-increment-MUTSPD-r10000d18000-052920-153932.h5f"]



	# sweep1 = h5py.File(sweepPath + sweepFiles[0], "r")
	good1 = h5py.File(goodPath + goodFiles[0], "r")

	# runPlotter(sweep1["1"])
	# fftPlotter(sweep1["1"])

	runPlotter(good1["1"])
	fftPlotter(good1["1"])



	# sweep1.close()
	good1.close()



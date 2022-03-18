'''TVG generator for FISC DAC Gain.
Usage: lutGen.py <Range-in-m> <Pre-amp-gain> <c> <fc>

Options:
    -h --help  Show this

'''
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

def TVG(Range=5, G_pre=10, c=1500, f_update=3200):
    VGA_Gres = 50 # 50dB per volt
    ## LO gain mode: G = 50 dB/V * Vgain - 6.50dB
    ## HI gain mode: G = 50 dB/V * Vgain + 5.5
        # VgainLO = (G+6.5) / 50
        # VgainHI = (G-5.5) / 50

    #c = 1500 ## Sound speed in water
    resolution = 1.214/1024 ## 10 bit DAC Vrange = 0-1.214V

    times = np.arange(0, Range*2.0/c, 1/f_update, dtype=np.float64)

    ranges = c*times
    gainArr = []
    for i, val in enumerate(ranges):
        if ranges[i] < 1:
            ranges[i] = 20.0*np.log10(1)#1
        else:
            gainArr[i] = 20.0*np.log10(ranges[i])

    #gainArr = G_pre + 20*np.log10(ranges)
    gainArr_length = len(gainArr)
    plt.plot(ranges, gainArr, label='iybbiy')
    plt.show()
    plt.legend()
    quit()

    #plt.plot(times, gainArr)
    #print("ranges", ranges, "gainArr", gainArr)
    #plt.title('TVG gain over time')
    #plt.text(2e-4, 12, 'N sampls for DAC: '+str(gainArr_length))
    #print('N sampls for DAC: ', str(gainArr_length))
    #plt.show()



def main():
   arguments= docopt(__doc__, version='FISC DAC lutGen scipt 1.0')
   print("test", arguments)
   Range = np.double(arguments['<Range-in-m>'])
   G_pre = np.double(arguments['<Pre-amp-gain>'])
   f_update = np.double(arguments['<fc>'])
   c = np.double(arguments['<c>'])
   genLUT(Range, G_pre, c, f_update)


if __name__ == '__main__':
    main()

#!/usr/bin/python3
"""
Add checksum to NMEA sentence, send over serial and echo reply back

Usage:
  nmea.py <device> <message>
  nmea.py -h | --help

Arguments:
  <device>        Serial device
  <message>       Message, without "$", "*", "\\r", "\\n" or checksum

Options:
  -h --help     Show this screen.

Try:
  nmea.py /dev/ttyUSB0 SOP,1,2
"""


from docopt import docopt
import serial
import operator
from functools import reduce


def checksum(sentence):
    sentence = sentence.strip(b'$\r\n')
    try:
        nmeadata, cksum = sentence.split(b'*', 1)
    except ValueError:
        nmeadata = sentence
        cksum = "0"

    calc_cksum = reduce(operator.xor, (s for s in nmeadata), 0)
    return nmeadata, int(cksum, 16), calc_cksum


if __name__ == '__main__':
    arguments = docopt(__doc__, version='NMEA.py 1.0')
    message = arguments['<message>']


    if any(ext in message for ext in ["\r", "\n", "$", "*"]):
        print("Don't include any of \"\\r\" \"\\n\" \"$\" or \"*\" in message")

    _, _, crc = checksum(message.encode())
    message = "${0}*{1:x}\r\n".format(message, crc)
    print("<-- {0}".format(message).strip('\r\n'))

    with serial.Serial(arguments['<device>'], 9600, timeout=1) as ser:
        ser.write(message.encode())
        packet = ser.readline()
        print("--> {0}".format(packet.decode().strip('\r\n')))
        message, cksum, calc_cksum = checksum(packet)
        if calc_cksum is not cksum:
            print("CRC NOT OK")

import numpy as np


rx_beamwidth = 32.25#15.68
tx_beamwidth = 12.07

A = (90-rx_beamwidth/2)
B = (90-tx_beamwidth/2)
C = (180 - A - B)

c = 0.3

print("Angle A:", A)
print("Angle B:", B)
print("Angle C:", C)
print("RX to TX distance:", c)

x = (c * np.sin(np.deg2rad(A)) * np.sin(np.deg2rad(B)) ) / (np.sin(np.deg2rad(C)))

print("Radial Acoustic Deadzone:", x)

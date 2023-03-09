from math import pi, sqrt, e

def gaussian(x, mean, std):
    return (1/(std*sqrt(2*pi))*e)**(-.5*((x-mean)/std)**2)
from random import choice
from numpy import array, dot, random
import numpy as np

step = lambda x: 0 if x < 0 else 1

data = [
    (array([1,0,0]), 0),
    (array([1,0,1]), 0),
    (array([1,1,0]), 0),
    (array([1,1,1]), 1),
]

n = 100
w = random.rand(3)
lr = 0.01
E=1

while E != 0:
    E=0
    for i in xrange(n):
        x, di = choice(data)
        yi = dot(w, x)
        ei = di - step(yi)
        w += lr * ei * x
        E += ei ** 2

for x,_ in data:
    yi = dot(x, w)
    print("{}: {} -> {}".format(x[1:3], yi, step(yi)))
    print("Weights: ")
    print(w)
    print("\n")









    

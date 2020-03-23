from numpy import array, dot, random
import numpy as np

def step(v):
    return 0 if v < 0 else 1

my_data = np.array([[1,1,0,1,1,1,1,1,0,1],[1,0,1,0,0,1,0,0,1,0]])
result = np.array([[1,0], [0,1]])

lr = 0.01
E=1
N,n = my_data.shape
w1 = np.random.randn(n, 1)
w2 = np.random.randn(n, 1)

while E != 0:
    E=0
    for i in range(N):
        yi1 = step(np.dot(my_data[i], w1))
        yi2 = step(np.dot(my_data[i], w2))
        ei1 = result[i][0] - yi1
        ei2 = result[i][1] - yi2
        w1 += lr * ei1 * my_data[i].reshape(n, 1)
        w2 += lr * ei2 * my_data[i].reshape(n, 1)
        E += ei1 ** 2
        E += ei2 ** 2

      
    print("{}: {} -> {}".format(my_data[0,:], yi1, step(yi1)))
    print("Weights: ")
    print(w1)
    print("\n")
    print("{}: {} -> {}".format(my_data[1,:], yi2, step(yi2)))
    print("Weights: ")
    print(w2)
    print("\n")

    


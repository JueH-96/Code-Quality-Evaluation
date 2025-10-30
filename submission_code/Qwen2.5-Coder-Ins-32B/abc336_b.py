# YOUR CODE HERE
import sys

def ctz(X):
    if X == 0:
        return 0
    count = 0
    while (X & 1) == 0:
        X >>= 1
        count += 1
    return count

N = int(sys.stdin.read().strip())
print(ctz(N))
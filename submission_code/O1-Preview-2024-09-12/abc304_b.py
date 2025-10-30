# YOUR CODE HERE
import sys
import math

N = int(sys.stdin.readline())
if N == 0:
    D = 0
else:
    D = max(0, int(math.log10(N)) - 2)
K = 10 ** D
M = (N // K) * K
print(M)
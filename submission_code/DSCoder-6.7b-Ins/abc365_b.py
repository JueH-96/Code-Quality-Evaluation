# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

A.sort()
print(A[-2])
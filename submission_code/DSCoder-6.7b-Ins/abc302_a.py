# YOUR CODE HERE

import sys

A, B = map(int, sys.stdin.readline().split())

if A % B == 0:
    print(A // B - 1)
else:
    print(A // B)
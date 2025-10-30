# YOUR CODE HERE
import sys

A = list(map(int, sys.stdin.readline().split()))
result = sum(A[i] * (2**i) for i in range(64))
print(result)
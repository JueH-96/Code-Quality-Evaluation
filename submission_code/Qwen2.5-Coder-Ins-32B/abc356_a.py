# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
L = int(input[1])
R = int(input[2])

A = list(range(1, N + 1))
A[L-1:R] = reversed(A[L-1:R])

print(" ".join(map(str, A)))
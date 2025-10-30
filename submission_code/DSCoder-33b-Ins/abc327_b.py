# YOUR CODE HERE
import math

B = int(input())

A = int(math.pow(B, 1/2))

if A**A == B:
    print(A)
else:
    print(-1)
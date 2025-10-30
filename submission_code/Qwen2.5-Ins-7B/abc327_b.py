# YOUR CODE HERE
import math

B = int(input())

for A in range(1, int(B**0.5) + 2):
    if A**A == B:
        print(A)
        break
else:
    print(-1)
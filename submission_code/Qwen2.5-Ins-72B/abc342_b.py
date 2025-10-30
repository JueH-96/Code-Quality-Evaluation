# YOUR CODE HERE
import sys

# Read input
N = int(input())
P = list(map(int, input().split()))
Q = int(input())

# Create a dictionary to map person number to their position
position = {P[i]: i + 1 for i in range(N)}

# Process queries
for _ in range(Q):
    A, B = map(int, input().split())
    if position[A] < position[B]:
        print(A)
    else:
        print(B)
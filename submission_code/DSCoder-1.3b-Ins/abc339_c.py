# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

passengers = 0
for i in range(N):
    passengers += A[i]
    if passengers < 0:
        passengers = 0

print(passengers)
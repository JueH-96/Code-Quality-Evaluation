import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Initialize the number of balls in each box
X = [0] * N

# Perform the operations
for i in range(M):
    C = 0
    box_to_take_from = B[i]
    while A[box_to_take_from] > 0:
        C += 1
        box_to_put_in = (box_to_take_from + C) % N
        X[box_to_put_in] += 1
        A[box_to_take_from] -= 1

# Print the final number of balls in each box
print(" ".join(map(str, X)))
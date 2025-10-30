import sys

# Read input
N = int(sys.stdin.readline().strip())
P = [int(x) for x in sys.stdin.readline().strip().split()]
A = [int(x) for x in sys.stdin.readline().strip().split()]

# Perform the operation until the lexicographically smallest A is obtained
while True:
    new_A = [0] * N
    for i in range(N):
        new_A[i] = A[P[i] - 1]
    if new_A < A:
        A = new_A
    else:
        break

# Print the result
print(" ".join(map(str, A)))
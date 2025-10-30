import sys

# Read the inputs
N = int(sys.stdin.readline().strip())
P = list(map(int, sys.stdin.readline().strip().split()))
Q = int(sys.stdin.readline().strip())

# Create a list to store the positions of the people
positions = [0] * (N+1)
for i in range(N):
    positions[P[i]] = i+1

# Process the queries
for _ in range(Q):
    A, B = map(int, sys.stdin.readline().strip().split())
    if positions[A] > positions[B]:
        print(A)
    else:
        print(B)
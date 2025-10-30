# YOUR CODE HERE

# Read the inputs
N = int(input())
P = list(map(int, input().split()))
Q = int(input())

# Create a list to store the position of each person
positions = [0] * (N + 1)
for i in range(N):
    positions[P[i]] = i + 1

# Process the queries
for _ in range(Q):
    A, B = map(int, input().split())
    if positions[A] < positions[B]:
        print(A)
    else:
        print(B)
import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Create array A with 1-based indexing (A[0] unused)
A = [0]  # Index 0 is unused
for _ in range(N):
    val = int(data[index])
    A.append(val)
    index += 1

# Create position array pos where pos[num] is the position of number num
pos = [0] * (N + 1)
for i in range(1, N + 1):  # i is position
    num = A[i]
    pos[num] = i

# List to store the swaps
swaps = []

# Fix each position from 1 to N-1
for i in range(1, N):  # From 1 to N-1 inclusive
    if A[i] != i:
        p = pos[i]  # Position where number i is located
        # Record the swap
        swaps.append((i, p))
        # Perform the swap on A
        X = A[i]  # Number currently at position i
        # Swap A[i] and A[p]
        A[i], A[p] = A[p], A[i]
        # Update position array
        pos[i] = i  # Number i is now at position i
        pos[X] = p  # Number X is now at position p

# Output the result
K = len(swaps)
print(K)
for swap in swaps:
    print(f"{swap[0]} {swap[1]}")
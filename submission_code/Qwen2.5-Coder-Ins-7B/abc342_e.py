import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Parse the input data
trains = []
index = 2
for _ in range(M):
    l = int(data[index])
    d = int(data[index + 1])
    k = int(data[index + 2])
    c = int(data[index + 3])
    A = int(data[index + 4])
    B = int(data[index + 5])
    trains.append((l, d, k, c, A, B))
    index += 6

# Initialize the f array with -infinity
f = [-float('inf')] * (N + 1)

# Process each train
for l, d, k, c, A, B in trains:
    for t in range(l, l + k * d, d):
        if f[A] != -float('inf'):
            f[B] = max(f[B], t + c)

# Output the results
for i in range(1, N):
    if f[i] == -float('inf'):
        print("Unreachable")
    else:
        print(f[i])
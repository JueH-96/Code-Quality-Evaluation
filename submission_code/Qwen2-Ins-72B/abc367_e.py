import sys
from collections import defaultdict

N, K = map(int, sys.stdin.readline().split())
X = list(map(int, sys.stdin.readline().split()))
A = list(map(int, sys.stdin.readline().split()))

# Create a dictionary to store the cycles
cycles = defaultdict(list)
visited = [False] * N

# Find cycles in X
for i in range(N):
    if not visited[i]:
        cycle = []
        j = i
        while not visited[j]:
            visited[j] = True
            cycle.append(j)
            j = X[j] - 1
        for index, val in enumerate(cycle):
            cycles[index].append(val)

# Apply operations for each cycle
for cycle in cycles.values():
    cycle_len = len(cycle)
    cycle_mod = K % cycle_len
    for i, idx in enumerate(cycle):
        new_idx = cycle[(i + cycle_mod) % cycle_len]
        A[idx] = A[new_idx]

print(*A)
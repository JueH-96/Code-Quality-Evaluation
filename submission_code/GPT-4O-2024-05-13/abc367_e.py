# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
X = list(map(int, data[2:N+2]))
A = list(map(int, data[N+2:]))

# Adjust X to be zero-indexed
X = [x - 1 for x in X]

# To handle the large K, we need to find the cycle in the permutation
visited = [False] * N
cycles = []

for i in range(N):
    if not visited[i]:
        cycle = []
        current = i
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = X[current]
        cycles.append(cycle)

# Apply the cycles to A
for cycle in cycles:
    cycle_length = len(cycle)
    effective_k = K % cycle_length
    if effective_k == 0:
        continue
    temp = [A[cycle[(i - effective_k) % cycle_length]] for i in range(cycle_length)]
    for i in range(cycle_length):
        A[cycle[i]] = temp[i]

print(' '.join(map(str, A)))
import sys
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
K = data[index]
index += 1
P = [0] * (N + 1)
for i in range(1, N + 1):
    P[i] = data[index]
    index += 1
visited = [False] * (N + 1)
cycles = []
for i in range(1, N + 1):
    if not visited[i]:
        cycle = []
        x = i
        while not visited[x]:
            visited[x] = True
            cycle.append(x)
            x = P[x]
        cycles.append(cycle)
result = [0] * (N + 1)
for cycle in cycles:
    L = len(cycle)
    R = pow(2, K, L)
    for j in range(L):
        pos = cycle[j]
        new_val = cycle[(j + R) % L]
        result[pos] = new_val
print(' '.join(map(str, [result[i] for i in range(1, N + 1)])))
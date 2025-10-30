# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = []
index = 2
for _ in range(M):
    u = int(data[index]) - 1
    v = int(data[index + 1]) - 1
    b = int(data[index + 2])
    c = int(data[index + 3])
    edges.append((u, v, b, c))
    index += 4

def can_achieve_ratio(ratio):
    dist = [-float('inf')] * N
    dist[0] = 0
    for _ in range(N):
        updated = False
        for u, v, b, c in edges:
            if dist[u] != -float('inf'):
                new_value = dist[u] + b - ratio * c
                if new_value > dist[v]:
                    dist[v] = new_value
                    updated = True
        if not updated:
            break
    return dist[N-1] >= 0

low, high = 0, 10000
while high - low > 1e-9:
    mid = (low + high) / 2
    if can_achieve_ratio(mid):
        low = mid
    else:
        high = mid

print(f"{low:.15f}")
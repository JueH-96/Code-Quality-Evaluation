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
    edges.append((u, v))
    index += 2

W = list(map(int, data[index:index + N]))
index += N
A = list(map(int, data[index:index + N]))

# Create adjacency list
adj = [[] for _ in range(N)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Priority queue to process vertices by their weight in descending order
pq = []
for i in range(N):
    if A[i] > 0:
        heapq.heappush(pq, (-W[i], i))

operations = 0

while pq:
    weight, x = heapq.heappop(pq)
    weight = -weight
    if A[x] == 0:
        continue

    A[x] -= 1
    operations += 1

    # Find the set S of adjacent vertices
    for y in adj[x]:
        if W[y] < weight:
            A[y] += 1
            heapq.heappush(pq, (-W[y], y))

print(operations)
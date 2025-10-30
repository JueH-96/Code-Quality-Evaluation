# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

index = 0

N = int(data[index])
index += 1
M = int(data[index])
index += 1
Q = int(data[index])
index += 1

edges = []
for _ in range(M):
    A = int(data[index]) - 1
    index += 1
    B = int(data[index]) - 1
    index += 1
    C = int(data[index])
    index += 1
    edges.append((A, B, C))

queries = []
for _ in range(Q):
    t = int(data[index])
    index += 1
    if t == 1:
        i = int(data[index]) - 1
        index += 1
        queries.append((1, i))
    else:
        x = int(data[index]) - 1
        index += 1
        y = int(data[index]) - 1
        index += 1
        queries.append((2, x, y))

# Initialize adjacency list
adj = [[] for _ in range(N)]
for i, (A, B, C) in enumerate(edges):
    adj[A].append((B, C, i))
    adj[B].append((A, C, i))

closed = set()

def dijkstra(start, end):
    dist = [float('inf')] * N
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, length, idx in adj[u]:
            if idx in closed:
                continue
            new_dist = current_dist + length
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return dist[end] if dist[end] != float('inf') else -1

results = []

for query in queries:
    if query[0] == 1:
        closed.add(query[1])
    else:
        x, y = query[1], query[2]
        results.append(dijkstra(x, y))

for result in results:
    print(result)
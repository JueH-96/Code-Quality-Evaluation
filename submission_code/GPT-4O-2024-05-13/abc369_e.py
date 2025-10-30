# YOUR CODE HERE
import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0

N = int(data[index])
index += 1
M = int(data[index])
index += 1

bridges = []
graph = defaultdict(list)

for i in range(M):
    U = int(data[index]) - 1
    index += 1
    V = int(data[index]) - 1
    index += 1
    T = int(data[index])
    index += 1
    bridges.append((U, V, T))
    graph[U].append((V, T, i))
    graph[V].append((U, T, i))

Q = int(data[index])
index += 1

queries = []
for _ in range(Q):
    K = int(data[index])
    index += 1
    B = [int(data[index + i]) - 1 for i in range(K)]
    index += K
    queries.append(B)

def dijkstra(start, end, required_bridges):
    pq = [(0, start, set())]
    visited = {}
    
    while pq:
        time, node, used_bridges = heapq.heappop(pq)
        
        if (node, tuple(sorted(used_bridges))) in visited:
            continue
        
        visited[(node, tuple(sorted(used_bridges)))] = time
        
        if node == end and all(b in used_bridges for b in required_bridges):
            return time
        
        for neighbor, travel_time, bridge_index in graph[node]:
            new_time = time + travel_time
            new_used_bridges = used_bridges | {bridge_index}
            heapq.heappush(pq, (new_time, neighbor, new_used_bridges))
    
    return float('inf')

results = []
for query in queries:
    result = dijkstra(0, N-1, set(query))
    results.append(result)

for result in results:
    print(result)
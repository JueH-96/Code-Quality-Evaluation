# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []

index = 1
for i in range(N-1):
    A = int(data[index])
    B = int(data[index+1])
    C = int(data[index+2])
    edges.append((C, A-1, B-1))
    index += 3

# Prim's algorithm to find the minimum spanning tree (MST)
def prim(N, edges):
    adj = [[] for _ in range(N)]
    for cost, u, v in edges:
        adj[u].append((cost, v))
        adj[v].append((cost, u))
    
    total_cost = 0
    visited = [False] * N
    min_heap = [(0, 0)]  # (cost, node)
    
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        total_cost += cost
        for next_cost, v in adj[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (next_cost, v))
    
    return total_cost

mst_cost = prim(N, edges)
# Since we need to visit all cities and return to the starting point,
# the minimum travel distance is twice the MST cost.
print(2 * mst_cost)
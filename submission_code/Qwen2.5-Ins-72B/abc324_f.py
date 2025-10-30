import sys
from heapq import heappop, heappush

def solve(N, M, edges):
    graph = [[] for _ in range(N)]
    for u, v, b, c in edges:
        graph[u-1].append((v-1, b, c))
    
    def check(x):
        dist = [float('inf')] * N
        dist[0] = 0
        pq = [(0, 0, 0, 0)]  # (cost, beauty, vertex, path_cost)
        
        while pq:
            cost, beauty, vertex, path_cost = heappop(pq)
            if vertex == N-1:
                return True
            if cost > dist[vertex]:
                continue
            for next_vertex, b, c in graph[vertex]:
                new_cost = cost + c
                new_beauty = beauty + b
                if new_cost - x * new_beauty < dist[next_vertex]:
                    dist[next_vertex] = new_cost - x * new_beauty
                    heappush(pq, (new_cost, new_beauty, next_vertex, path_cost + c))
        
        return False
    
    low, high = 0, 10000
    while high - low > 1e-9:
        mid = (low + high) / 2
        if check(mid):
            low = mid
        else:
            high = mid
    
    return low

# Read input
input = sys.stdin.read
data = input().split()
N, M = int(data[0]), int(data[1])
edges = [(int(data[2*i+2]), int(data[2*i+3]), int(data[2*i+4]), int(data[2*i+5])) for i in range(M)]

# Solve and print the result
result = solve(N, M, edges)
print(f"{result:.10f}")
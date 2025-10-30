import sys
from heapq import heappush, heappop

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n, m = int(data[0]), int(data[1])
    edges = [[] for _ in range(n)]
    idx = 2
    for _ in range(m):
        u, v, b, c = map(int, (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]))
        u, v = u - 1, v - 1
        edges[u].append((v, b, c))
        idx += 4
    
    # Binary search on the answer
    low, high = 0.0, 1e9
    while high - low > 1e-9:
        mid = (low + high) / 2
        if can_reach(n, edges, mid):
            low = mid
        else:
            high = mid
    
    print(low)

def can_reach(n, edges, ratio):
    # Dijkstra's algorithm to find the shortest path with the given ratio
    dist = [float('inf')] * n
    dist[0] = 0
    pq = [(0, 0)]  # (cost, node)
    
    while pq:
        cost, u = heappop(pq)
        if u == n - 1:
            return True
        if cost > dist[u]:
            continue
        for v, b, c in edges[u]:
            new_cost = cost + c - b / ratio
            if new_cost < dist[v]:
                dist[v] = new_cost
                heappush(pq, (new_cost, v))
    
    return False

solve()
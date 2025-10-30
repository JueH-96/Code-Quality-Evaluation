import heapq
from collections import defaultdict

def solve():
    N, M, X = map(int, input().split())
    
    # Build adjacency lists for both original and reversed graphs
    graph = defaultdict(list)
    reversed_graph = defaultdict(list)
    
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        reversed_graph[v].append(u)
    
    # Dijkstra's algorithm with states (vertex, direction)
    # direction: 0 = original graph, 1 = reversed graph
    dist = {}
    dist[(1, 0)] = 0
    
    pq = [(0, 1, 0)]  # (cost, vertex, direction)
    
    while pq:
        cost, v, dir = heapq.heappop(pq)
        
        # Skip if we've already found a better path
        if (v, dir) in dist and dist[(v, dir)] < cost:
            continue
        
        # If we reached vertex N, return the cost
        if v == N:
            print(cost)
            return
        
        # Try moving along edges in current direction
        current_graph = graph if dir == 0 else reversed_graph
        for next_v in current_graph[v]:
            new_cost = cost + 1
            if (next_v, dir) not in dist or dist[(next_v, dir)] > new_cost:
                dist[(next_v, dir)] = new_cost
                heapq.heappush(pq, (new_cost, next_v, dir))
        
        # Try reversing all edges
        new_dir = 1 - dir
        new_cost = cost + X
        if (v, new_dir) not in dist or dist[(v, new_dir)] > new_cost:
            dist[(v, new_dir)] = new_cost
            heapq.heappush(pq, (new_cost, v, new_dir))

solve()
from collections import defaultdict
from heapq import heappush, heappop

def solve():
    # Read input
    N, M, X = map(int, input().split())
    
    # Create adjacency lists for both original and reversed edges
    graph = defaultdict(list)
    rev_graph = defaultdict(list)
    
    # Store edges
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        rev_graph[v].append(u)
    
    # Initialize distance array for all states
    # dist[v][is_reversed] represents minimum cost to reach vertex v in state is_reversed
    dist = {}
    for i in range(1, N+1):
        dist[(i, False)] = float('inf')
        dist[(i, True)] = float('inf')
    
    # Start from vertex 1 with cost 0
    dist[(1, False)] = 0
    
    # Priority queue for Dijkstra's algorithm
    # Format: (cost, vertex, is_reversed)
    pq = [(0, 1, False)]
    
    while pq:
        cost, v, is_reversed = heappop(pq)
        
        # If this state has been processed with a better cost, skip
        if cost > dist[(v, is_reversed)]:
            continue
        
        # If we reached the target vertex N, this is the minimum cost
        if v == N:
            print(cost)
            return
        
        # Try moving along edges in current state
        curr_graph = rev_graph if is_reversed else graph
        for next_v in curr_graph[v]:
            new_cost = cost + 1
            if new_cost < dist[(next_v, is_reversed)]:
                dist[(next_v, is_reversed)] = new_cost
                heappush(pq, (new_cost, next_v, is_reversed))
        
        # Try reversing all edges
        new_is_reversed = not is_reversed
        new_cost = cost + X
        if new_cost < dist[(v, new_is_reversed)]:
            dist[(v, new_is_reversed)] = new_cost
            heappush(pq, (new_cost, v, new_is_reversed))

solve()
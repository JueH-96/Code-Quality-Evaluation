import heapq
from collections import defaultdict
import sys

def solve():
    N, M, X = map(int, input().split())
    
    # Build adjacency lists for both orientations
    graph = [defaultdict(list), defaultdict(list)]  # graph[0] = original, graph[1] = reversed
    
    for _ in range(M):
        u, v = map(int, input().split())
        graph[0][u].append(v)  # original: u -> v
        graph[1][v].append(u)  # reversed: v -> u
    
    # Dijkstra with states (cost, vertex, orientation)
    # orientation: 0 = original, 1 = reversed
    pq = [(0, 1, 0)]  # start at vertex 1 with original orientation
    dist = {}
    
    while pq:
        cost, vertex, orientation = heapq.heappop(pq)
        
        if (vertex, orientation) in dist:
            continue
            
        dist[(vertex, orientation)] = cost
        
        if vertex == N:
            print(cost)
            return
        
        # Option 1: Move along an edge (cost +1)
        for next_vertex in graph[orientation][vertex]:
            if (next_vertex, orientation) not in dist:
                heapq.heappush(pq, (cost + 1, next_vertex, orientation))
        
        # Option 2: Reverse all edges (cost +X)
        new_orientation = 1 - orientation
        if (vertex, new_orientation) not in dist:
            heapq.heappush(pq, (cost + X, vertex, new_orientation))

solve()
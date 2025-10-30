# YOUR CODE HERE
import sys
from collections import defaultdict
from heapq import heappush, heappop

def dijkstra(graph, start, end, must_use):
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start, frozenset())]
    
    while pq:
        dist, node, used = heappop(pq)
        
        if node == end and used == must_use:
            return dist
        
        for neighbor, weight, bridge in graph[node]:
            new_dist = dist + weight
            new_used = used | {bridge} if bridge in must_use else used
            
            if len(new_used) > len(must_use):
                continue
            
            if new_dist < distances[(node, new_used)]:
                distances[(node, new_used)] = new_dist
                heappush(pq, (new_dist, neighbor, new_used))
    
    return float('inf')

def solve():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    
    for i in range(1, M + 1):
        u, v, t = map(int, input().split())
        graph[u].append((v, t, i))
        graph[v].append((u, t, i))
    
    Q = int(input())
    
    for _ in range(Q):
        K = int(input())
        bridges = set(map(int, input().split()))
        result = dijkstra(graph, 1, N, frozenset(bridges))
        print(result)

solve()
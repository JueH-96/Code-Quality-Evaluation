# YOUR CODE HERE
import sys
from collections import defaultdict
from heapq import heappop, heappush

def solve():
    input = sys.stdin.read
    data = input().split()
    n, i = 0, 0
    nodes = defaultdict(list)
    while i < len(data):
        if data[i] == ' ':
            n = int(data[i+1])
            i += 2
        else:
            l, r, c = int(data[i]), int(data[i+1]), int(data[i+2])
            nodes[n+l].append((n+r, c))
            nodes[n+r].append((n+l, c))
            i += 3
    
    visited = [False] * (n + 1)
    pq = [(0, 1)]
    total_cost = 0
    while pq:
        cost, node = heappop(pq)
        if visited[node]:
            continue
        visited[node] = True
        total_cost += cost
        for neighbor, edge_cost in nodes[node]:
            if not visited[neighbor]:
                heappush(pq, (edge_cost, neighbor))
    
    if all(visited):
        print(total_cost)
    else:
        print(-1)

solve()
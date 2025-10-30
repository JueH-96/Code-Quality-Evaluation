# YOUR CODE HERE
import sys
from collections import defaultdict

def solve(N1, N2, M, edges):
    # Create adjacency lists for the two sets of vertices
    adj1 = defaultdict(set)
    adj2 = defaultdict(set)
    
    for a, b in edges:
        if 1 <= a <= N1 and 1 <= b <= N1:
            adj1[a].add(b)
            adj1[b].add(a)
        elif N1 + 1 <= a <= N1 + N2 and N1 + 1 <= b <= N1 + N2:
            adj2[a].add(b)
            adj2[b].add(a)
        elif 1 <= a <= N1 and N1 + 1 <= b <= N1 + N2:
            adj1[a].add(b)
            adj2[b].add(a)
        elif 1 <= b <= N1 and N1 + 1 <= a <= N1 + N2:
            adj1[b].add(a)
            adj2[a].add(b)
    
    # Find the shortest path from 1 to N1+N2 after adding an edge
    def bfs(start, adj):
        visited = set()
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node == N1 + N2:
                return dist
            if node not in visited:
                visited.add(node)
                for neighbor in adj[node]:
                    queue.append((neighbor, dist + 1))
        return float('inf')
    
    # Find the maximum possible d
    max_d = 0
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N1 + N2 + 1):
            if v not in adj1[u]:
                adj1[u].add(v)
                adj2[v].add(u)
                d = bfs(1, adj1)
                max_d = max(max_d, d)
                adj1[u].remove(v)
                adj2[v].remove(u)
    
    return max_d + 1

# Read input
input = sys.stdin.read
N1, N2, M, *edges = map(int, input().split())
edges = list(zip(edges[::2], edges[1::2]))

# Solve and print the result
print(solve(N1, N2, M, edges))
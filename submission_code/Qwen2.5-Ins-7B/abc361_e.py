# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    N = int(input())
    graph = defaultdict(list)
    for _ in range(N-1):
        A, B, C = map(int, input().split())
        graph[A-1].append((B-1, C))
        graph[B-1].append((A-1, C))
    
    def dfs(node, parent, dist):
        max_dist = 0
        for neighbor, weight in graph[node]:
            if neighbor != parent:
                max_dist = max(max_dist, dfs(neighbor, node, weight))
        return max_dist + dist
    
    max_dist_from_root = dfs(0, -1, 0)
    print((max_dist_from_root + 1) * 2)

solve()
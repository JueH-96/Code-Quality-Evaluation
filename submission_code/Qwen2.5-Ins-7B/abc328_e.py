# YOUR CODE HERE
from typing import List
from heapq import heappop, heappush
import sys

input = sys.stdin.readline

def kruskal(n: int, edges: List[List[int]], k: int) -> int:
    parent = list(range(n))
    rank = [0] * n
    edges.sort(key=lambda x: x[2])
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] > rank[py]:
            parent[py] = px
        elif rank[px] < rank[py]:
            parent[px] = py
        else:
            parent[py] = px
            rank[px] += 1
        return True
    
    cost = 0
    for u, v, w in edges:
        if union(u-1, v-1):
            cost = (cost + w) % k
            if len(edges) == n-1:
                break
    return cost

n, m, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append([u, v, w])

print(kruskal(n, edges, k))
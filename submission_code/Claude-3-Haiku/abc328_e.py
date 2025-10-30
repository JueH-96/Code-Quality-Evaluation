import heapq

def kruskal(n, m, k, edges):
    parent = list(range(n+1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    edges.sort(key=lambda x: x[2])
    total_cost = 0
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            total_cost = (total_cost + w) % k
    
    return total_cost

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    
    print(kruskal(n, m, k, edges))

solve()
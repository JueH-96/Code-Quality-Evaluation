from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

def can_satisfy(people, n):
    uf = UnionFind(n + 1)
    
    # First pass: process equality constraints
    for s, t in people:
        uf.union(s, t)
    
    # Second pass: build directed graph of inequality constraints
    graph = defaultdict(set)
    for s, t in people:
        if s < t:
            for k in range(s + 1, t):
                graph[uf.find(s)].add(uf.find(k))
        else:
            for k in range(t + 1, s):
                graph[uf.find(k)].add(uf.find(s))
    
    # Check for cycles using DFS
    def has_cycle():
        color = {}  # 0: white, 1: gray, 2: black
        
        def dfs(node):
            if node in color:
                return color[node] == 1  # Gray means we're in the current DFS path
            color[node] = 1
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            color[node] = 2
            return False
        
        for node in graph:
            if node not in color and dfs(node):
                return True
        return False
    
    return not has_cycle()

# Read input
n, m, q = map(int, input().split())
people = []
for _ in range(m):
    s, t = map(int, input().split())
    people.append((s, t))

# Process queries
for _ in range(q):
    l, r = map(int, input().split())
    if can_satisfy(people[l-1:r], n):
        print("Yes")
    else:
        print("No")
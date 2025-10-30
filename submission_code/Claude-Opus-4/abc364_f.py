# YOUR CODE HERE
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True
    
    def is_connected(self):
        return self.components == 1

N, Q = map(int, input().split())

# Collect all edges
edges = []
for i in range(Q):
    L, R, C = map(int, input().split())
    hub_vertex = N + i
    for j in range(L-1, R):  # L-1 to R-1 for 0-indexed
        edges.append((C, hub_vertex, j))

# Sort edges by cost
edges.sort()

# Kruskal's algorithm
uf = UnionFind(N + Q)
total_cost = 0
edges_used = 0

for cost, u, v in edges:
    if uf.union(u, v):
        total_cost += cost
        edges_used += 1
        if edges_used == N + Q - 1:  # Found spanning tree
            break

if uf.is_connected():
    print(total_cost)
else:
    print(-1)
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

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

k = int(input())
forbidden_pairs = []
for _ in range(k):
    x, y = map(int, input().split())
    forbidden_pairs.append((x, y))

q = int(input())

# Build Union-Find structure
uf = UnionFind(n + 1)
for u, v in edges:
    uf.union(u, v)

# Process queries
for _ in range(q):
    p, q_val = map(int, input().split())
    root_p = uf.find(p)
    root_q = uf.find(q_val)
    
    if root_p == root_q:
        # Already connected, graph remains good
        print("Yes")
    else:
        # Check if connecting these components connects any forbidden pair
        good = True
        for x, y in forbidden_pairs:
            root_x = uf.find(x)
            root_y = uf.find(y)
            
            # If one is in comp_p and the other is in comp_q, they'll be connected
            if (root_x == root_p and root_y == root_q) or (root_x == root_q and root_y == root_p):
                good = False
                break
        
        print("Yes" if good else "No")
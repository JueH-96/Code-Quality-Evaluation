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
    
    def same(self, x, y):
        return self.find(x) == self.find(y)

# Read input
N, M = map(int, input().split())

uf = UnionFind(N)

for _ in range(M):
    u, v = map(int, input().split())
    u -= 1  # 0-indexed
    v -= 1
    uf.union(u, v)

K = int(input())
constraints = []
for _ in range(K):
    x, y = map(int, input().split())
    x -= 1  # 0-indexed
    y -= 1
    constraints.append((x, y))

Q = int(input())
for _ in range(Q):
    p, q = map(int, input().split())
    p -= 1  # 0-indexed
    q -= 1
    
    good = True
    for x, y in constraints:
        # Check if adding edge (p, q) would connect x and y
        if (uf.same(x, p) and uf.same(y, q)) or (uf.same(x, q) and uf.same(y, p)):
            good = False
            break
    
    print("Yes" if good else "No")
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

n, q = map(int, input().split())
uf = UnionFind(n)

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Type 1: Add edge
        u, v = query[1] - 1, query[2] - 1  # Convert to 0-indexed
        uf.union(u, v)
    
    else:  # Type 2: Find k-th largest
        v, k = query[1] - 1, query[2]  # Convert v to 0-indexed
        
        # Find all vertices in the same component as v
        root = uf.find(v)
        connected = []
        
        for i in range(n):
            if uf.find(i) == root:
                connected.append(i + 1)  # Convert back to 1-indexed
        
        # Sort in descending order
        connected.sort(reverse=True)
        
        if k <= len(connected):
            print(connected[k - 1])
        else:
            print(-1)
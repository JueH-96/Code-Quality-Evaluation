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

N, Q = map(int, input().split())
uf = UnionFind(N)

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1] - 1, query[2] - 1
        uf.union(u, v)
    else:
        v, k = query[1] - 1, query[2]
        root_v = uf.find(v)
        
        component = []
        for i in range(N):
            if uf.find(i) == root_v:
                component.append(i + 1)
        
        component.sort(reverse=True)
        
        if k <= len(component):
            print(component[k - 1])
        else:
            print(-1)
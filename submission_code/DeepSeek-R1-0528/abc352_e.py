import sys

class UnionFind:
    __slots__ = ('parent', 'rank')
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    ops = []
    for _ in range(m):
        k = int(next(it))
        c = int(next(it))
        arr = [int(next(it)) for _ in range(k)]
        ops.append((c, k, arr))
    
    ops.sort(key=lambda x: x[0])
    
    uf = UnionFind(n)
    comp_count = n
    total_weight = 0
    
    for c, k, arr in ops:
        comp_set = set()
        for node in arr:
            idx = node - 1
            comp_set.add(uf.find(idx))
        
        k_comp = len(comp_set)
        total_weight += (k_comp - 1) * c
        if k_comp > 1:
            comp_list = list(comp_set)
            base = comp_list[0]
            for j in range(1, k_comp):
                uf.union(base, comp_list[j])
            comp_count -= (k_comp - 1)
    
    print(total_weight if comp_count == 1 else -1)

if __name__ == "__main__":
    main()
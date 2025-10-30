import sys
import threading

def main():
    import sys
    from itertools import combinations
    
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    edges = []
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        edges.append((u, v, w))
    
    # Union-Find / Disjoint Set
    class DSU:
        def __init__(self, n):
            self.p = list(range(n))
            self.r = [0]*n
            self.count = n
        def find(self, x):
            while self.p[x] != x:
                self.p[x] = self.p[self.p[x]]
                x = self.p[x]
            return x
        def unite(self, a, b):
            ra = self.find(a)
            rb = self.find(b)
            if ra == rb:
                return False
            if self.r[ra] < self.r[rb]:
                ra, rb = rb, ra
            self.p[rb] = ra
            if self.r[ra] == self.r[rb]:
                self.r[ra] += 1
            self.count -= 1
            return True
    
    best = None
    # Enumerate all choices of N-1 edges
    idxs = range(M)
    need = N-1
    for comb in combinations(idxs, need):
        dsu = DSU(N)
        s = 0
        ok = True
        # Add edges, break if cycle
        for i in comb:
            u, v, w = edges[i]
            if not dsu.unite(u, v):
                ok = False
                break
            s += w
            # optional: s %= K  # can mod at end
        if not ok or dsu.count != 1:
            continue
        cost = s % K
        if best is None or cost < best:
            best = cost
            # theoretical minimum is 0; can break early if found
            if best == 0:
                break
    
    # Output the minimum cost mod K
    print(best)

if __name__ == "__main__":
    main()
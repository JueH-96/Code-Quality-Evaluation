def main():
    import sys
    from itertools import combinations

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    edges = []
    idx = 3
    for _ in range(M):
        u = int(data[idx]) - 1
        v = int(data[idx+1]) - 1
        w = int(data[idx+2])
        idx += 3
        edges.append((u, v, w))

    # If there's exactly N-1 edges and the graph is connected, there's only one spanning tree.
    if M == N - 1:
        # Sum all edge weights and take mod K
        tot = sum(w for _, _, w in edges) % K
        print(tot)
        return

    # A small Union-Find (Disjoint Set) implementation
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0]*n
            self.count = n  # number of connected components

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            self.count -= 1
            return True

    best_cost = None
    # Try all combinations of edges of size N-1
    for subset in combinations(range(M), N-1):
        uf = UnionFind(N)
        cycle_found = False
        s = 0
        for eid in subset:
            u, v, w = edges[eid]
            if not uf.union(u, v):
                cycle_found = True
                break
            s += w
        if not cycle_found and uf.count == 1:
            cost = s % K
            if best_cost is None or cost < best_cost:
                best_cost = cost
            # If we ever hit 0, we cannot do better than that.
            if best_cost == 0:
                break

    print(best_cost)

# Do not forget to call main()
if __name__ == "__main__":
    main()
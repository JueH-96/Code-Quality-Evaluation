def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # A fast iterator over the input data
    idx = 0
    def input():
        nonlocal idx
        val = input_data[idx]
        idx += 1
        return val

    # Union-Find (Disjoint Set) implementation
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0]*n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def unite(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                if self.rank[rx] < self.rank[ry]:
                    self.parent[rx] = ry
                elif self.rank[rx] > self.rank[ry]:
                    self.parent[ry] = rx
                else:
                    self.parent[ry] = rx
                    self.rank[rx] += 1

    # Read N, M
    N = int(input())
    M = int(input())

    uf = UnionFind(N)
    edges = []
    for _ in range(M):
        u = int(input()) - 1
        v = int(input()) - 1
        # Union in DS
        uf.unite(u, v)

    # Read K
    K = int(input())
    
    # We'll store all "conflict edges" in a set of pairs of components
    conflict = set()
    for _ in range(K):
        x = int(input()) - 1
        y = int(input()) - 1
        # Find their components
        cx = uf.find(x)
        cy = uf.find(y)
        # They must be different components (since there's no path)
        # Record that these two components should NEVER be merged
        if cx > cy:
            cx, cy = cy, cx
        conflict.add((cx, cy))

    # Read Q
    Q = int(input())

    # For each query, just check if the two endpoints belong
    # to the same component or if their pair of components is conflicted.
    out = []
    for _ in range(Q):
        p = int(input()) - 1
        q = int(input()) - 1
        cp = uf.find(p)
        cq = uf.find(q)
        if cp == cq:
            # Same component => no problem
            out.append("Yes")
        else:
            # Different components => check if there's a conflict
            if cp > cq:
                cp, cq = cq, cp
            if (cp, cq) in conflict:
                out.append("No")
            else:
                out.append("Yes")

    print("
".join(out))

# Let's call solve() to execute when this file is run standalone
def main():
    solve()

if __name__ == "__main__":
    main()
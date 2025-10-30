import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin

    # Read N, M
    N, M = map(int, data.readline().split())
    parent = list(range(N+1))
    rank = [0] * (N+1)

    # DSU find
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # DSU union
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        else:
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

    # Read edges and build DSU for G
    for _ in range(M):
        u, v = map(int, data.readline().split())
        # union even if u==v or duplicate
        union(u, v)

    # Compress all roots
    for i in range(1, N+1):
        parent[i] = find(i)

    # Read K forbidden pairs
    K = int(data.readline())
    # We'll build a map from component -> set of forbidden other components
    forbid = {}
    for _ in range(K):
        x, y = map(int, data.readline().split())
        rx = parent[x]
        ry = parent[y]
        # They must differ by problem statement
        # Add both directions
        if rx not in forbid:
            forbid[rx] = set()
        forbid[rx].add(ry)
        if ry not in forbid:
            forbid[ry] = set()
        forbid[ry].add(rx)

    # Read Q queries
    Q = int(data.readline())
    out = []
    for _ in range(Q):
        p, q = map(int, data.readline().split())
        rp = parent[p]
        rq = parent[q]
        # If same component, no new connections among distinct components
        if rp == rq:
            out.append("Yes")
        else:
            # If merging these connects any forbidden pair?
            # That happens iff rq is in forbid[rp]
            if rp in forbid and rq in forbid[rp]:
                out.append("No")
            else:
                out.append("Yes")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
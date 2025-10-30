import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.readline
    # Read N, M
    line = data().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1])
    # DSU
    parent = list(range(N+1))
    rank = [0]*(N+1)
    def find(x):
        # iterative path-compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        else:
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

    # Read M edges
    for _ in range(M):
        u,v = map(int, data().split())
        # union even if u==v (loop) or duplicate, union handles it
        union(u, v)

    # Flatten all roots once
    for v in range(1, N+1):
        parent[v] = find(v)

    # Read K forbidden pairs
    K = int(data())
    forbidden = set()
    for _ in range(K):
        x,y = map(int, data().split())
        rx = parent[x]
        ry = parent[y]
        if rx == ry:
            # According to problem, this shouldn't happen
            continue
        if rx < ry:
            forbidden.add((rx, ry))
        else:
            forbidden.add((ry, rx))

    # Read Q queries
    Q = int(data())
    out = []
    for _ in range(Q):
        p,q = map(int, data().split())
        rp = parent[p]
        rq = parent[q]
        if rp == rq:
            # No new merge, original was good
            out.append("Yes")
        else:
            a, b = (rp, rq) if rp < rq else (rq, rp)
            if (a, b) in forbidden:
                out.append("No")
            else:
                out.append("Yes")

    # Output
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # Group testimonies by A: for each A, list of (B, D)
    byA = [[] for _ in range(N+1)]
    testimonies = []
    for _ in range(M):
        A, B, C = map(int, input().split())
        byA[A].append((B, C))
        testimonies.append((A, B, C))
    # DSU with parity: parent[x]<0 => root with size -parent[x]
    # d[x] = XOR from x to its root: R[x] XOR R[root] = d[x]
    parent = [-1] * (N+1)
    d = [0] * (N+1)
    def find(x):
        if parent[x] < 0:
            return x
        px = parent[x]
        root = find(px)
        d[x] ^= d[px]
        parent[x] = root
        return root
    def union(x, y, w):
        # enforce R[x] XOR R[y] = w
        rx = find(x)
        ry = find(y)
        if rx == ry:
            # check consistency
            if (d[x] ^ d[y]) != w:
                return False
            return True
        # join smaller into larger
        if parent[rx] > parent[ry]:
            rx, ry = ry, rx
            x, y = y, x
        # now rx has larger size
        # we set parent[rx] += parent[ry], parent[ry]=rx
        # need d[ry] so that for any y: new d[y] = old d[y]^d_add,
        # and then we satisfy R[x]^R[y] = w.
        # derivation gives d_add = w ^ d[x] ^ d[y]
        add = w ^ d[x] ^ d[y]
        parent[rx] += parent[ry]
        parent[ry] = rx
        d[ry] = add
        return True

    # Build XOR constraints among R variables:
    # For each A with multiple testimonies, pick first as reference
    ok = True
    for u in range(1, N+1):
        lst = byA[u]
        if len(lst) <= 1:
            continue
        v0, d0 = lst[0]
        for (v, dval) in lst[1:]:
            w = dval ^ d0  # we need R[v] XOR R[v0] = dval XOR d0
            if not union(v, v0, w):
                ok = False
                break
        if not ok:
            break

    if not ok:
        print(-1)
        return

    # Determine R[x] for each x: R[root]=0, R[x]=d[x]
    # Ensure we compress all
    for x in range(1, N+1):
        find(x)
    R = [0] * (N+1)
    for x in range(1, N+1):
        # now parent[x]<0 means root, d[x]=0
        R[x] = d[x]  # since we set R[root]=0

    # Compute C[u] for each u
    C = [0] * (N+1)
    for u in range(1, N+1):
        lst = byA[u]
        if not lst:
            C[u] = 0
        else:
            v0, d0 = lst[0]
            # R[u] XOR R[v0] = C[u] XOR d0  => C[u] = R[u] XOR R[v0] XOR d0
            C[u] = R[u] ^ R[v0] ^ d0

    # Output confusion bits C[1..N]
    out = ''.join('1' if C[i] else '0' for i in range(1, N+1))
    print(out)

if __name__ == "__main__":
    main()
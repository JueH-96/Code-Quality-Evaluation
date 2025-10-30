import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    # Precompute base cost = sum of shortest distances on the cycle
    base = 0
    # We'll build a difference array D[1..N+1] for the N bridges.
    # Bridge i is between islands i and i+1 mod N (bridge N is between N and 1).
    D = [0] * (N+2)
    for i in range(M-1):
        u = X[i]
        v = X[i+1]
        # cw is #steps going u->u+1->...->v, mod N
        cw = (v - u) % N
        ccw = N - cw
        w = min(cw, ccw)
        z = max(cw, ccw)
        base += w
        delta = z - w
        if delta == 0:
            continue
        # Determine which edges the *shortest* path uses.
        # If cw <= ccw, the shortest path goes cw: edges u,u+1,...,v-1 (mod N).
        # Otherwise it goes ccw, which in the cw-indexing is edges v,v+1,...,u-1.
        if cw <= ccw:
            # interval [u .. v-1] modulo N
            l = u
            r = (v - 1) if v-1 >= 1 else N
        else:
            # interval [v .. u-1] modulo N
            l = v
            r = (u - 1) if u-1 >= 1 else N
        # Now add delta to D[l..r] in circular sense
        if l <= r:
            D[l] += delta
            D[r+1] -= delta
        else:
            # wrap: [l..N] and [1..r]
            D[l] += delta
            D[N+1] -= delta
            D[1] += delta
            D[r+1] -= delta

    # Accumulate difference array to get extra[e] for each bridge e.
    best_extra = 10**30
    cur = 0
    for e in range(1, N+1):
        cur += D[e]
        if cur < best_extra:
            best_extra = cur

    print(base + best_extra)

if __name__ == "__main__":
    main()
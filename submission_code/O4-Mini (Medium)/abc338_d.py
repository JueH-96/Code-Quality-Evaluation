import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    X = [int(next(it)) for _ in range(M)]
    # base cost: sum of minimal distances between consecutive X's
    base = 0
    # difference array for extra deltas on edges 1..N
    # we'll use indices 1..N+1 in D
    D = [0] * (N + 2)
    for i in range(M - 1):
        u = X[i]
        v = X[i+1]
        # forward (clockwise) distance from u to v
        df = (v - u + N) % N
        dc = N - df
        # choose minimal direction
        if df <= dc:
            md = df
            # cw shortest: interval of edges [u, v-1] mod N
            # endpoints for the interval
            l = u
            # compute v-1 mod N
            r = v - 1 if v > 1 else N
        else:
            md = dc
            # ccw shortest: interval of edges [v, u-1] mod N
            l = v
            r = u - 1 if u > 1 else N
        base += md
        # extra cost if this interval is broken
        delta = N - 2 * md
        if delta == 0:
            continue
        # add delta to D[l..r] cyclically
        if l <= r:
            D[l] += delta
            D[r+1] -= delta
        else:
            # wrap around: [l..N] and [1..r]
            D[l] += delta
            D[N+1] -= delta
            D[1] += delta
            D[r+1] -= delta

    # build prefix sums to get extra cost on each edge
    best_extra = 10**30
    cur = 0
    # edges are 1..N
    for e in range(1, N+1):
        cur += D[e]
        if cur < best_extra:
            best_extra = cur

    # total best = base + minimal extra
    ans = base + best_extra
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()
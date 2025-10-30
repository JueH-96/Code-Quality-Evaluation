import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    mod = 998244353

    N = int(input())
    P = [0] * (N - 1)
    Q = [0] * (N - 1)
    for i in range(N - 1):
        p, q = map(int, input().split())
        P[i] = p
        Q[i] = q

    parent = list(range(N + 1))
    dist = [0] * (N + 1)   # dist[x]: accumulated offset from x to its root
    tag = [0] * (N + 1)    # tag[r]: total added to whole set at root r
    size = [1] * (N + 1)   # size[r]: size of set with root r

    def find(x):
        if parent[x] != x:
            p = parent[x]
            r = find(p)
            dist[x] = (dist[x] + dist[p]) % mod
            parent[x] = r
        return parent[x]

    def union(ru, rv):
        # merge set rv into ru (we assume ru and rv are roots)
        # After merging, parent[rv]=ru and set dist[rv] so that tags remain consistent
        parent[rv] = ru
        # we want tag[rv] + dist[rv] == tag[ru]
        # so dist[rv] = tag[rv] - tag[ru]
        diff = tag[rv] - tag[ru]
        diff %= mod
        dist[rv] = diff
        size[ru] += size[rv]

    def add_to_set(x, v):
        # add v to all members in set containing x
        rx = find(x)
        tag[rx] = (tag[rx] + v) % mod

    def modinv(x):
        return pow(x, mod - 2, mod)

    # Process each match
    for i in range(N - 1):
        u = P[i]
        v = Q[i]
        ru = find(u)
        rv = find(v)
        a = size[ru]
        b = size[rv]
        tot = a + b
        inv_tot = modinv(tot)
        # Prob ru wins = a / (a+b), rv wins = b / (a+b)
        pwU = a * inv_tot % mod
        pwV = b * inv_tot % mod
        # add expected win to all in ru and rv
        tag[ru] = (tag[ru] + pwU) % mod
        tag[rv] = (tag[rv] + pwV) % mod
        # merge smaller into larger
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        union(ru, rv)

    # Compute answers
    res = [0] * (N + 1)
    for i in range(1, N + 1):
        ri = find(i)
        # total for i = tag[ri] + dist[i]
        ans = tag[ri] + dist[i]
        res[i] = ans % mod

    # Output
    out = " ".join(str(res[i]) for i in range(1, N + 1))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()
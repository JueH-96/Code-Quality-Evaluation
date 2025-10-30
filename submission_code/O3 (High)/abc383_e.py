import sys


class DSU:
    def __init__(self, n, diff):
        self.parent = list(range(n + 1))
        self.sz = [1] * (n + 1)
        self.diff = diff  # diff[v] : (#A  - #B) still unmatched in the component whose root is v

    def find(self, v):
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def union(self, a, b, w):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return 0  # no new pairs formed

        da = self.diff[ra]
        db = self.diff[rb]

        # pairs which can be matched right now
        pairs = 0
        if da > 0 and db < 0:
            pairs = min(da, -db)
        elif da < 0 and db > 0:
            pairs = min(-da, db)

        # union by size
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
            da, db = db, da   # they have been swapped

        # attach rb under ra
        self.parent[rb] = ra
        self.sz[ra] += self.sz[rb]
        self.diff[ra] = da + db  # remaining unmatched items in merged component
        return pairs * w


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    idx = 0
    N = data[idx]; idx += 1
    M = data[idx]; idx += 1
    K = data[idx]; idx += 1

    edges = []
    for _ in range(M):
        u = data[idx]; v = data[idx + 1]; w = data[idx + 2]
        idx += 3
        edges.append((w, u, v))

    A = data[idx: idx + K]; idx += K
    B = data[idx: idx + K]

    # diff[v] = (#A occurrences) - (#B occurrences) for vertex v
    diff = [0] * (N + 1)
    for a in A:
        diff[a] += 1
    for b in B:
        diff[b] -= 1

    dsu = DSU(N, diff)

    edges.sort()   # sort by weight ascending

    ans = 0
    for w, u, v in edges:
        ans += dsu.union(u, v, w)

    print(ans)


if __name__ == "__main__":
    main()
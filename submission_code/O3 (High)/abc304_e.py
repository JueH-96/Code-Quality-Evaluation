import sys


class DSU:
    """ Disjoint Set Union (Union-Find) structure """
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def main() -> None:
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    n = int(next(it))
    m = int(next(it))

    dsu = DSU(n)

    # build the original graph
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        dsu.union(u, v)

    # forbidden pairs
    k = int(next(it))
    forbidden = set()
    for _ in range(k):
        x = int(next(it)) - 1
        y = int(next(it)) - 1
        rx, ry = dsu.find(x), dsu.find(y)
        if rx == ry:        # guaranteed not to happen, but keep it safe
            continue
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))

    # queries
    q = int(next(it))
    out = []
    for _ in range(q):
        p = int(next(it)) - 1
        qv = int(next(it)) - 1
        rp, rq = dsu.find(p), dsu.find(qv)
        if rp == rq:
            out.append("Yes")
        else:
            a, b = (rp, rq) if rp < rq else (rq, rp)
            out.append("No" if (a, b) in forbidden else "Yes")

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()
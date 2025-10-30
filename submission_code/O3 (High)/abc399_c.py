import sys

class DSU:
    """Disjoint Set Union – keeps track of connected components."""
    __slots__ = ('parent', 'size')

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        """Path–compressed find."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Union by size."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    it = iter(sys.stdin.read().split())
    n = int(next(it))
    m = int(next(it))

    dsu = DSU(n)

    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        dsu.union(u, v)

    # Count connected components
    roots = {dsu.find(i) for i in range(n)}
    c = len(roots)

    # Minimum deletions = M - (N - C)
    answer = m - (n - c)
    print(answer)


if __name__ == "__main__":
    main()
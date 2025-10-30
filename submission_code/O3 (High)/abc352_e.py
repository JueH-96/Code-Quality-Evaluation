import sys

class DSU:
    __slots__ = ("parent", "size")
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size   = [1]*n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x]  += self.size[y]
        return True


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it   = iter(data)
    N    = next(it)            # number of vertices
    M    = next(it)            # number of operations

    ops  = []
    for _ in range(M):
        K = next(it)           # size of subset
        C = next(it)           # weight of every edge in the clique
        vertices = [next(it) for _ in range(K)]
        ops.append((C, vertices))

    # Kruskal: process groups (cliques) in non-decreasing weight order
    ops.sort(key=lambda x: x[0])

    dsu   = DSU(N + 1)         # vertices are 1-based
    total = 0

    for w, verts in ops:
        roots = set()
        for v in verts:
            roots.add(dsu.find(v))
        cnt = len(roots)
        if cnt > 1:
            total += (cnt - 1) * w
            itr = iter(roots)
            first_root = next(itr)
            for r in itr:
                dsu.union(first_root, r)

    # check connectivity
    root0 = dsu.find(1)
    for v in range(2, N + 1):
        if dsu.find(v) != root0:
            print(-1)
            return
    print(total)


if __name__ == "__main__":
    main()
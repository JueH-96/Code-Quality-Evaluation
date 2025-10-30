import sys
sys.setrecursionlimit(1 << 25)


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    dsu = DSU(N)

    for _ in range(M):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        dsu.union(a, b)

    # Aggregate sizes of components
    root_sizes = {}
    for v in range(N):
        r = dsu.find(v)
        root_sizes[r] = root_sizes.get(r, 0) + 1

    total_possible_edges = 0
    for sz in root_sizes.values():
        total_possible_edges += sz * (sz - 1) // 2

    answer = total_possible_edges - M
    print(answer)


if __name__ == "__main__":
    main()
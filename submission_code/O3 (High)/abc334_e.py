import sys

MOD = 998244353


class DSU:
    """Disjoint Set Union (Union-Find)"""
    __slots__ = ("parent", "size")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]

    n = h * w
    dsu = DSU(n)

    # 1. build connected components of existing green cells (‘#’)
    for i in range(h):
        for j in range(w):
            if grid[i][j] != '#':
                continue
            idx = i * w + j
            if i + 1 < h and grid[i + 1][j] == '#':        # down
                dsu.union(idx, (i + 1) * w + j)
            if j + 1 < w and grid[i][j + 1] == '#':        # right
                dsu.union(idx, i * w + (j + 1))

    # 2. count initial number of green components (C)
    comp_roots = set()
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                comp_roots.add(dsu.find(i * w + j))
    C = len(comp_roots)

    # 3. for each red cell (‘.’) compute k = #distinct neighbouring green components
    red_cnt = 0            # R
    sum_k = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                red_cnt += 1
                neighbours = set()
                if i > 0 and grid[i - 1][j] == '#':
                    neighbours.add(dsu.find((i - 1) * w + j))
                if i + 1 < h and grid[i + 1][j] == '#':
                    neighbours.add(dsu.find((i + 1) * w + j))
                if j > 0 and grid[i][j - 1] == '#':
                    neighbours.add(dsu.find(i * w + (j - 1)))
                if j + 1 < w and grid[i][j + 1] == '#':
                    neighbours.add(dsu.find(i * w + (j + 1)))
                sum_k += len(neighbours)

    # 4. Expected value  E = C + 1 − (1/R) * Σ k(c)
    #    So numerator P = (C + 1) * R − Σ k(c), denominator Q = R
    P = ((C + 1) * red_cnt - sum_k) % MOD
    inv_R = pow(red_cnt, MOD - 2, MOD)        # modular inverse of the denominator
    answer = (P * inv_R) % MOD
    print(answer)


if __name__ == "__main__":
    main()
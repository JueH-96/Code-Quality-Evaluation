from typing import List, Tuple

class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.parents = [-1] * n

    def find(self, x: int) -> int:
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x: int, y: int) -> None:
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x: int) -> int:
        return -self.parents[self.find(x)]

    def same(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def members(self, x: int) -> List[int]:
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self) -> List[int]:
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self) -> int:
        return len(self.roots())

    def all_group_members(self) -> List[List[int]]:
        return [self.members(r) for r in self.roots()]

    def __str__(self) -> str:
        return "
".join("{}: {}".format(r, self.members(r)) for r in self.roots())

def kruskal(n: int, edges: List[Tuple[int, int, int]]) -> int:
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(n)
    cost = 0
    for u, v, w in edges:
        if not uf.same(u, v):
            uf.union(u, v)
            cost += w
    return cost if uf.group_count() == 1 else -1

def main():
    n, m = map(int, input().split())
    edges = []
    uf = UnionFind(n)
    for _ in range(m):
        k, c = map(int, input().split())
        a = list(map(int, input().split()))
        for i in range(k):
            a[i] -= 1
        for i in range(k):
            for j in range(i + 1, k):
                if not uf.same(a[i], a[j]):
                    edges.append((a[i], a[j], c))
                    uf.union(a[i], a[j])
    if uf.group_count() > 1:
        print(-1)
        return
    print(kruskal(n, edges))

if __name__ == "__main__":
    main()
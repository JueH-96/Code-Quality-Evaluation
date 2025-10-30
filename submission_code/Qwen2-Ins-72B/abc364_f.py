from typing import List, Tuple
from collections import defaultdict
from heapq import heappush, heappop

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
        return '
'.join(f'{r}: {m}' for r, m in zip(self.roots(), self.all_group_members()))

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
    N, Q = map(int, input().split())
    edges = []
    uf = UnionFind(N + Q)
    for i in range(Q):
        L, R, C = map(int, input().split())
        edges.append((N + i, L - 1, C))
        edges.append((N + i, R, C))
        uf.union(L - 1, R)

    for i in range(Q):
        L, R, C = map(int, input().split())
        if not uf.same(L - 1, R):
            edges.append((L - 1, R, 0))

    print(kruskal(N + Q, edges))

if __name__ == "__main__":
    main()
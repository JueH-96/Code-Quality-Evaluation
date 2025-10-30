# YOUR CODE HERE
import sys
from typing import List, Tuple
from heapq import heappush, heappop

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        self.size -= 1
        return True

def kruskal(n: int, edges: List[Tuple[int, int, int]]) -> int:
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    total_cost = 0
    for u, v, cost in edges:
        if uf.union(u, v):
            total_cost += cost
    return total_cost if uf.size == 1 else -1

def main():
    N, Q = map(int, input().split())
    edges = []
    for i in range(Q):
        L, R, C = map(int, input().split())
        for j in range(L - 1, R):
            edges.append((N + i, j, C))

    result = kruskal(N + Q, edges)
    print(result)

if __name__ == "__main__":
    main()
from collections import defaultdict, deque
from typing import List

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        return {r: self.members(r) for r in self.roots()}

    def __str__(self):
        return '
'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

def main():
    N, M = map(int, input().split())
    uf = UnionFind(N)
    edges = []
    has_1 = [False] * N
    has_M = [False] * N
    for i in range(N):
        A = int(input())
        S = list(map(int, input().split()))
        if 1 in S:
            has_1[i] = True
        if M in S:
            has_M[i] = True
        for j in range(A):
            S[j] -= 1
        for j in range(A):
            for k in range(j + 1, A):
                edges.append((S[j], S[k] + 1, i))
    edges.sort()
    groups_1 = set()
    groups_M = set()
    for u, v, w in edges:
        if uf.same(w, groups_1) and uf.same(w, groups_M):
            break
        if has_1[w]:
            groups_1 = uf.find(w)
        if has_M[w]:
            groups_M = uf.find(w)
        uf.union(w, v)
    if not uf.same(groups_1, groups_M):
        print(-1)
        return
    ans = 0
    for i in range(N):
        if uf.find(i) == i and not (has_1[i] and has_M[i]):
            ans += 1
    print(ans)

if __name__ == '__main__':
    main()
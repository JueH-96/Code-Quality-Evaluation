from collections import defaultdict

class UnionFind:
    def __init__(self, n):
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

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def size(self, x):
        return -self.parents[self.find(x)]

    def members(self, x):
        root = self.find(x)
        return [i for i in range(len(self.parents)) if self.find(i) == root]

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
    uf = UnionFind(N + 1)

    for _ in range(M):
        a, b = map(int, input().split())
        uf.union(a, b)

    K = int(input())
   禁止s = [0] * (N + 1)
    for _ in range(K):
        x, y = map(int, input().split())
        if uf.same(x, y):
            continue
        禁止x = min(uf.find(x), uf.find(y))
        禁止y = max(uf.find(x), uf.find(y))
        禁止s[禁止x] |= 1 << (禁止y - 1)

    Q = int(input())
    for _ in range(Q):
        p, q = map(int, input().split())
        禁止p = uf.find(p)
        禁止q = uf.find(q)
        if 禁止p > 禁止q:
            禁止p, 禁止q = 禁止q, 禁止p

        if (禁止s[禁止p] >> (禁止q - 1)) & 1 == 0:
            print('Yes')
        else:
            print('No')

if __name__ == "__main__":
    main()
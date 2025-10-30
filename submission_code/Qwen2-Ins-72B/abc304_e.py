from collections import deque, defaultdict

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
    edges = [list(map(int, input().split())) for _ in range(M)]
    K = int(input())
    pairs = [list(map(int, input().split())) for _ in range(K)]
    Q = int(input())
    queries = [list(map(int, input().split())) for _ in range(Q)]

    uf = UnionFind(N)
    for u, v in edges:
        uf.union(u-1, v-1)

    pair_to_root = defaultdict(set)
    for x, y in pairs:
        root_x = uf.find(x-1)
        root_y = uf.find(y-1)
        pair_to_root[root_x].add(root_y)
        pair_to_root[root_y].add(root_x)

    for p, q in queries:
        root_p = uf.find(p-1)
        root_q = uf.find(q-1)
        if root_p == root_q:
            print("No")
        elif root_q in pair_to_root[root_p] or root_p in pair_to_root[root_q]:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()
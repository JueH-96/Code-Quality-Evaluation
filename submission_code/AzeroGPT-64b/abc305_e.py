from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.par = list(range(n))
        self.rank = [0] * n
        self.group_size = [1] * n
        self.groups = n

    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return

        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.group_size[y] += self.group_size[x]
        else:
            self.par[y] = x
            self.group_size[x] += self.group_size[y]
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1
        self.groups -= 1

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def size(self, x):
        return self.group_size[self.find(x)]

    def all_groups(self):
        leader = [self.find(i) for i in range(self.n)]
        result = dict()
        for i in range(self.n):
            if leader[i] not in result:
                result[leader[i]] = []
            result[leader[i]].append(i)
        return result

def solve(N, M, K, guards):
    uf = UnionFind(N + 1)
    distance = [0] * (N + 1)

    for a, b in guards:
        uf.union(a, b)

    for i in range(1, N + 1):
        leader = uf.find(i)
        if leader != i:
            distance[i] = distance[leader] + 1
        else:
            distance[i] = 0

    guarded = set()
    for p, h in guards:
        leader = uf.find(p)
        for i, d in enumerate(distance):
            if uf.find(i) == leader and d <= h:
                guarded.add(i)

    return guarded

N, M, K = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(M)]
guards = [list(map(int, input().split())) for _ in range(K)]

result = solve(N, M, K, edges)

print(len(result))
print(' '.join(map(str, sorted(result))))
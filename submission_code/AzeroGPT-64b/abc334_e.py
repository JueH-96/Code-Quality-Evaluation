H, W = map(int, input().split())
S = [list(input()) for _ in range(H)]

mod = 998244353

class UnionFind():
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
        dict_members = {r: self.members(r) for r in self.roots()}
        return dict_members

    def __str__(self):
        return '
'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

uf = UnionFind(H * W)
ans = 0
cnt = 0
for i in range(H):
    for j in range(W):
        if S[i][j] == "#":
            uf.union(i*W+j, i*W+j)
            cnt += 1
            if i > 0 and S[i-1][j] == "#":
                    uf.union(i*W+j, (i-1)*W+j)
            if j > 0 and S[i][j-1] == "#":
                    uf.union(i*W+j, i*W+j-1)

flat = uf.all_group_members()
total = [0] * (H*W)
for a in flat.values():
    for t in a:
        total[uf.find(t)] += 1
# print(flat)
# print(total)


dp = [0] * (H*W + 1)
dp[0] = 1
for val in total:
    for i in range(H*W, val-1, -1):
        dp[i] = dp[i] + dp[i - val]
        dp[i] %= mod
# print(dp)

inv2 = pow(2, mod - 2, mod)

for i in range(H):
    for j in range(W):
        cnts = 0
        if S[i][j] == ".": 
            if i > 0 and S[i-1][j] == "#":
                cnts += total[uf.find((i-1) * W + j)]
            if j > 0 and S[i][j-1] == "#":
                cnts += total[uf.find(i*W+j-1)]
            if i < H-1 and S[i+1][j] == "#":
                cnts += total[uf.find((i+1)*W+j)]
            if j < W-1 and S[i][j+1] == "#":
                cnts += total[uf.find(i*W+j+1)]

            ans += dp[cnt + 1] * pow(2, cnts, mod) * pow(2, mod - 2, mod) % mod
            ans %= mod

print(ans*inv2 % mod)
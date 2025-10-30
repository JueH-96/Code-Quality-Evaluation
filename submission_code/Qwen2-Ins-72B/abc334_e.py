from collections import defaultdict, deque
import sys
input = sys.stdin.readline

MOD = 998244353

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
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return '
'.join(f'{r}: {m}' for r, m in self.all_group_members().items())

def main():
    H, W = map(int, input().split())
    S = [list(input().strip()) for _ in range(H)]
    uf = UnionFind(H * W)
    cnt = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#':
                cnt[i][j] = 1
                for dx, dy in [(0, 1), (1, 0)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W and S[nx][ny] == '#':
                        uf.union(i * W + j, nx * W + ny)
    uf2 = UnionFind(H * W)
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.':
                for dx, dy in [(0, 1), (1, 0)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W and S[nx][ny] == '#':
                        uf2.union(i * W + j, nx * W + ny)
    ans = 0
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.':
                tmp = 0
                for dx, dy in [(0, 1), (1, 0)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W and S[nx][ny] == '#':
                        tmp += uf.size(nx * W + ny)
                ans += tmp * uf2.size(i * W + j)
    ans = ans * pow(H * W - sum(map(sum, cnt)), MOD - 2, MOD)
    ans = ans * sum(map(sum, cnt)) % MOD
    ans = (ans + sum(uf.size(i * W + j) for i in range(H) for j in range(W) if S[i][j] == '#')) % MOD
    print(ans)

if __name__ == '__main__':
    main()
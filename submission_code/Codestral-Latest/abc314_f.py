from collections import defaultdict
import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
matches = [(int(data[i]), int(data[i+1])) for i in range(1, 2*n, 2)]

MOD = 998244353

def mod_inverse(a, m):
    return pow(a, m-2, m)

def solve(n, matches):
    parent = list(range(n+1))
    size = [1] * (n+1)
    wins = [0] * (n+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if size[rootX] > size[rootY]:
                rootX, rootY = rootY, rootX
            parent[rootX] = rootY
            size[rootY] += size[rootX]

    for p, q in matches:
        rootP = find(p)
        rootQ = find(q)
        prob_win_p = size[rootP] / (size[rootP] + size[rootQ])
        prob_win_q = size[rootQ] / (size[rootP] + size[rootQ])

        wins[p] += prob_win_p
        wins[q] += prob_win_q

        union(p, q)

    expected_wins = [0] * (n+1)
    for i in range(1, n+1):
        expected_wins[i] = wins[i]

    return expected_wins[1:]

expected_wins = solve(n, matches)
result = [int(x * mod_inverse(1, MOD)) % MOD for x in expected_wins]
print(*result)
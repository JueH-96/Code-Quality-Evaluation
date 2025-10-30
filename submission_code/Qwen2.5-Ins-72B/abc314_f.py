# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def modinv(x, mod=MOD):
    return pow(x, mod-2, mod)

def solve(n, matches):
    parent = list(range(n + 1))
    size = [1] * (n + 1)
    win_expectation = [0] * (n + 1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
            size[root_x] += size[root_y]
            return root_x, root_y
        return None, None

    for p, q in matches:
        root_p, root_q = union(p, q)
        if root_p is not None and root_q is not None:
            a = size[root_p]
            b = size[root_q]
            prob_p = a * modinv(a + b, MOD) % MOD
            prob_q = b * modinv(a + b, MOD) % MOD
            win_expectation[root_p] = (win_expectation[root_p] + prob_p) % MOD
            win_expectation[root_q] = (win_expectation[root_q] + prob_q) % MOD

    for i in range(1, n + 1):
        root = find(i)
        win_expectation[i] = (win_expectation[i] + win_expectation[root]) % MOD

    return win_expectation[1:]

n = int(input())
matches = [tuple(map(int, input().split())) for _ in range(n - 1)]

result = solve(n, matches)
print(' '.join(map(str, result)))
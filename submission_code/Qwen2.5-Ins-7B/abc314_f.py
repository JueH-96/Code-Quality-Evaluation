# YOUR CODE HERE
from collections import defaultdict
from math import gcd

def modular_inverse(x, mod):
    def extended_gcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, z = extended_gcd(b % a, a)
            return (g, z - (b // a) * y, y)
    _, inv, _ = extended_gcd(x, mod)
    return inv % mod

def solve():
    mod = 998244353
    n = int(input())
    edges = [tuple(map(int, input().split())) for _ in range(n-1)]
    graph = defaultdict(list)
    for u, v in edges:
        graph[u-1].append(v-1)
        graph[v-1].append(u-1)
    
    dp = [[0, 1] for _ in range(n)]
    for i in range(n):
        dp[i][1] = pow(sum(1 for j in graph[i] if j < i), -1, mod)
    
    stack = [0]
    while stack:
        u = stack.pop()
        for v in graph[u]:
            if v < u:
                continue
            dp[v][0] += dp[u][0] * dp[u][1] * dp[v][1] % mod
            dp[v][1] *= dp[u][1] * (dp[u][0] + dp[u][1]) % mod
            dp[v][1] %= mod
            stack.append(v)
    
    for i in range(n):
        print(dp[i][0] * modular_inverse(dp[i][1], mod) % mod, end=' ')
    print()

solve()
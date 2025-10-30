# YOUR CODE HERE
import sys
input = sys.stdin.readline
mod = 998244353
N = int(input())
G = [[] for _ in range(N+1)]
for _ in range(N-1):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)
fact = [1]*(N+1)
inv = [1]*(N+1)
for i in range(1, N+1):
    fact[i] = i*fact[i-1]%mod
    inv[i] = pow(fact[i], mod-2, mod)
size = [0]*(N+1)
dp = [0]*(N+1)
def dfs(v, p=-1):
    size[v] = 1
    dp[v] = 1
    for u in G[v]:
        if u == p:
            continue
        dfs(u, v)
        size[v] += size[u]
        dp[v] = dp[v]*dp[u]%mod*inv[size[u]]%mod
    dp[v] = dp[v]*fact[size[v]-1]%mod
dfs(1)
ans = [0]*(N+1)
def reroot(v, p=-1):
    ans[v] = dp[v]*pow(dp[p]*inv[size[v]]%mod*fact[size[p]-size[v]]%mod, mod-2, mod)*fact[size[p]-1]%mod
    for u in G[v]:
        if u == p:
            continue
        dp[v], dp[u] = dp[u]*fact[size[u]]%mod*pow(dp[v]*inv[size[u]]%mod*fact[size[v]-1]%mod, mod-2, mod)*fact[size[v]]%mod, dp[v]
        size[v], size[u] = size[u], size[v]
        reroot(u, v)
        dp[v], dp[u] = dp[u]*fact[size[u]]%mod*pow(dp[v]*inv[size[u]]%mod*fact[size[v]-1]%mod, mod-2, mod)*fact[size[v]]%mod, dp[v]
        size[v], size[u] = size[u], size[v]
reroot(1)
print(*ans[1:])
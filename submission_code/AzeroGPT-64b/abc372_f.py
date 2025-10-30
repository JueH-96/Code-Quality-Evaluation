import numpy as np
MOD = 998244353

T = np.arange(200010, dtype=np.int64)
T[1:] = np.cumprod(T[1:], dtype=np.int64) % MOD

def modpowm(a, n, pm):
    y = a ** (n & -n) % pm
    if y==0:
        return y
    n >>= 1
    if n:
        x = modpowm(a, n, pm)
        return x * x * y % pm
    return y * y % pm

def inv(x):
    return modpowm(x, MOD - 2, MOD)

def modcomb(n, r):
    num = T[n]
    den = T[r] * T[n - r] % MOD
    return num * inv(den) % MOD

N, M, K = map(int, input().split())
G = [[] for _ in range(N)]
rev = [[] for _ in range(N)]
for i in range(M):
    u, v = map(int, input().split())
    u, v = u - 1, v - 1
    G[u].append(v)
    rev[v].append(u)

def dfs(v):
    Q = [v]
    P = [1]
    L = [0]
    while Q:
        y = Q.pop()
        py = P.pop()
        ly = L.pop()
        if vis[y] is not None:
            continue
        vis[y] = ly
        for x in G[y]:
            if vis[x] is None:
                Q.append(x)
                P.append(py)
                L.append(ly)
            elif vis[x] > ly:
                cands.append((y, x))
                vis[x] = ly
dfs(0)
vis = [None] * N
cands = []

tot = 0
for x,y in cands:
    if vis[x] is not None:
        if vis[y] is None or vis[x] < vis[y]:
            cyc = []
            n = x
            while n not in cyc:
                cyc.append(n)
                n = y
                y = x
            cyc.append(n)
            cyc.reverse()
            ans = 1
            st = [cyc[0]]
            check = set(cyc)
            sz = {}
            loop = []
            while st:
                n = st.pop()
                if n in sz:
                    continue
                sz[n] = len(loop)
                for c in G[n]:
                    rev[c].remove(n)
                    if c not in check:
                        loop.append(c)
                        st.append(c)
            cyc.reverse()
            for n in cyc:
                x = len(G[n]) + len(rev[n])
                y = len(loop) - sz[n]
                if y > 0:
                    ans *= modcomb(x, y)
                else:
                    ans *= modpowm(x, MOD - 2, MOD)
                ans %= MOD
                for c in G[n]:
                    if c not in check:
                        for pre in rev[c]:
                            rev[pre].remove(c)
                        rev[c] = []
            tot += ans
            tot %= MOD
if not cands:
    ans = modpowm(1, K, MOD)
else:
    ans = modcomb(N + K - 1, K)

chk = modpowm(1, K % MOD, MOD) * tot % MOD
print((ans - chk) % MOD)
MOD = 998244353

def pow_mod(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def inv_mod(x, p):
    return pow_mod(x, p - 2, p)

N, M = map(int, input().split())
total_nodes = N * M

E = [[] for _ in range(total_nodes + 1)]
for i in range(1, total_nodes + 1):
    E[i].append(max(i - N, 0))
    E[max(i - N, 0)].append(i)

dp = {}
def solve(u, painted):
    if (u, painted) in dp:
        return dp[(u, painted)]
    
    if bin(painted).count('1') == total_nodes:
        return 0

    ans = 0
    neighbors = E[u]
    for v in neighbors:
        new_painted = painted
        if v != 0 and (painted & (1 << v)) == 0:
            new_painted |= (1 << v)
        ans = (ans + solve(v, new_painted) + 1) % MOD
    
    ans = (ans * inv_mod(len(neighbors), MOD)) % MOD
    dp[(u, painted)] = ans
    return ans

print(solve(0, 1))
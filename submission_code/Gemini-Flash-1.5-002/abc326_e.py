# YOUR CODE HERE
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

n = int(input())
a = list(map(int, input().split()))

e = [0] * (n + 1)
for i in range(1, n + 1):
    e[i] = a[i - 1]
    for j in range(1, i):
        e[i] = (e[i] + e[j] * inv_mod(i, MOD)) % MOD

ans = 0
for i in range(1, n + 1):
    ans = (ans + e[i]) % MOD

print(ans)
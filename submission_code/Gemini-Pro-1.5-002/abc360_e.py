# YOUR CODE HERE
def power(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def modInverse(n, p):
    return power(n, p - 2, p)

n, k = map(int, input().split())
mod = 998244353

inv_n = modInverse(n, mod)
inv_n_sq = (inv_n * inv_n) % mod

dp = [0] * (n + 1)
dp[1] = 1

for i in range(k):
    new_dp = [0] * (n + 1)
    for j in range(1, n + 1):
        new_dp[j] = (dp[j] * (1 - inv_n - inv_n + inv_n_sq)) % mod
        if j > 1:
            new_dp[j] = (new_dp[j] + dp[j-1] * inv_n) % mod
        if j < n:
            new_dp[j] = (new_dp[j] + dp[j+1] * inv_n) % mod
    dp = new_dp

ans = 0
for i in range(1, n + 1):
    ans = (ans + dp[i] * i) % mod

print(ans)
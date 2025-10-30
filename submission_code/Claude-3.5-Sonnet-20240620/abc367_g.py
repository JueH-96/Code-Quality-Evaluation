# YOUR CODE HERE
MOD = 998244353

def pow_mod(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

N, M, K = map(int, input().split())
A = list(map(int, input().split()))

dp = [0] * M
dp[0] = 1

xor_sum = [0] * M
for i in range(N):
    new_dp = dp[:]
    for j in range(M):
        new_j = (j + 1) % M
        new_dp[new_j] = (new_dp[new_j] + dp[j]) % MOD
        xor_sum[new_j] = xor_sum[new_j] ^ (xor_sum[j] ^ A[i])
    dp = new_dp

result = 0
for i in range(M):
    if i == 0:
        count = (pow_mod(2, N, MOD) - 1 + MOD) % MOD
    else:
        count = dp[i]
    result = (result + count * pow_mod(xor_sum[i], K, MOD)) % MOD

print(result)
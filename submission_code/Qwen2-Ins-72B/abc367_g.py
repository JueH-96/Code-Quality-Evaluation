MOD = 998244353

def pow_mod(x, y, z):
    res = 1
    x %= z
    while y > 0:
        if y & 1:
            res = res * x % z
        y >>= 1
        x = x * x % z
    return res

def main():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    dp = [[0] * 21 for _ in range(21)]
    dp[0][0] = 1
    
    for i in range(20):
        for j in range(21):
            dp[i + 1][j] = dp[i][j] * 2 % MOD
            if j > 0:
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j - 1]) % MOD
    
    ans = 0
    xor_counts = [0] * (1 << 20)
    for a in A:
        xor_counts[a] += 1
    
    for xor in range(1 << 20):
        count = xor_counts[xor]
        if count == 0:
            continue
        score = pow_mod(xor, K, MOD)
        for i in range(1, N + 1):
            if i % M == 0:
                ans = (ans + dp[N - 1][i] * score * count % MOD) % MOD
            else:
                ans = (ans - dp[N - 1][i] * score * count % MOD) % MOD
        ans = (ans + MOD) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()
def main():
    N, X, *T = map(int, open(0).read().split())
    MOD = 998244353
    total_time = sum(T)
    dp = [0] * (X + 1)
    dp[0] = pow(total_time, MOD-2, MOD)
    for x in range(X):
        dp[x+1] = (dp[x] * total_time - dp[x] + dp[x] * dp[x+1]) * pow(total_time + 1, MOD-2, MOD) % MOD
    
    x = pow(total_time, MOD-2, MOD)
    y = dp[X] * pow(T[0] * x, MOD-2, MOD) % MOD * T[0] % MOD
    print(y)

if __name__ == '__main__':
    main()
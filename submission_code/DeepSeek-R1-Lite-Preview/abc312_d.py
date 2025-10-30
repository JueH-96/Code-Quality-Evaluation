def main():
    MOD = 998244353
    S = input().strip()
    n = len(S)
    dp = [0] * (n + 1)
    dp[0] = 1
    for c in S:
        dp_next = [0] * (n + 1)
        for j in range(n + 1):
            if dp[j] == 0:
                continue
            if c == '(':
                if j + 1 <= n:
                    dp_next[j + 1] = (dp_next[j + 1] + dp[j]) % MOD
            elif c == ')':
                if j > 0:
                    dp_next[j - 1] = (dp_next[j - 1] + dp[j]) % MOD
            elif c == '?':
                if j + 1 <= n:
                    dp_next[j + 1] = (dp_next[j + 1] + dp[j]) % MOD
                if j > 0:
                    dp_next[j - 1] = (dp_next[j - 1] + dp[j]) % MOD
        dp = dp_next
    print(dp[0] % MOD)

if __name__ == "__main__":
    main()
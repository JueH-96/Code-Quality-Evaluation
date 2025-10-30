def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    S = data.strip()
    n = len(S)
    MOD = 998244353

    # dp[j] = number of ways for current prefix to have balance j
    dp = [0] * (n + 2)
    dp[0] = 1  # empty prefix with balance 0

    for ch in S:
        # prepare next DP row
        ndp = [0] * (n + 2)
        if ch == '(':
            # only '(' increases balance
            for j in range(n):
                v = dp[j]
                if v:
                    ndp[j + 1] = (ndp[j + 1] + v) % MOD
        elif ch == ')':
            # only ')' decreases balance
            for j in range(1, n + 1):
                v = dp[j]
                if v:
                    ndp[j - 1] = (ndp[j - 1] + v) % MOD
        else:
            # '?': either '(' or ')'
            # treat as '('
            for j in range(n):
                v = dp[j]
                if v:
                    ndp[j + 1] = (ndp[j + 1] + v) % MOD
            # treat as ')'
            for j in range(1, n + 1):
                v = dp[j]
                if v:
                    ndp[j - 1] = (ndp[j - 1] + v) % MOD
        dp = ndp

    # answer is ways to end with balance 0
    print(dp[0] % MOD)

if __name__ == "__main__":
    main()
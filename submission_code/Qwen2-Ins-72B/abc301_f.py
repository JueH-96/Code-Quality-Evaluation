from collections import Counter

MOD = 998244353

def solve(S):
    dp = [0] * 16
    dp[0] = 1
    for s in S:
        ndp = [0] * 16
        if s == '?':
            for i in range(16):
                for j in range(26):
                    ndp[i | (j << 2)] = (ndp[i | (j << 2)] + dp[i]) % MOD
                for j in range(26):
                    ndp[i | (j << 4)] = (ndp[i | (j << 4)] + dp[i]) % MOD
        elif s.isupper():
            for i in range(16):
                ndp[i | (ord(s) - ord('A') << 2)] = (ndp[i | (ord(s) - ord('A') << 2)] + dp[i]) % MOD
                ndp[i | (ord(s) - ord('A') << 4)] = (ndp[i | (ord(s) - ord('A') << 4)] + dp[i]) % MOD
        else:
            for i in range(16):
                ndp[i | (ord(s) - ord('a') << 3)] = (ndp[i | (ord(s) - ord('a') << 3)] + dp[i]) % MOD
        dp = ndp
    return (sum(dp) - dp[15]) % MOD

S = input()
print(solve(S))
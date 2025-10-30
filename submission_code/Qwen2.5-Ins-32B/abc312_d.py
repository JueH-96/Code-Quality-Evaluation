import sys

def solve(s):
    MOD = 998244353
    n = len(s)
    dp = [[0 for _ in range(n+1)] for _ in range(n+1)]
    dp[0][0] = 1
    for c in s:
        new_dp = [[0 for _ in range(n+1)] for _ in range(n+1)]
        for bal in range(n+1):
            for open_count in range(n+1):
                if c == '(':
                    if bal < n:
                        new_dp[bal+1][open_count+1] += dp[bal][open_count]
                        new_dp[bal+1][open_count+1] %= MOD
                elif c == ')':
                    if bal > 0:
                        new_dp[bal-1][open_count] += dp[bal][open_count]
                        new_dp[bal-1][open_count] %= MOD
                else:  # '?'
                    if bal < n:
                        new_dp[bal+1][open_count+1] += dp[bal][open_count]
                        new_dp[bal+1][open_count+1] %= MOD
                    if bal > 0:
                        new_dp[bal-1][open_count] += dp[bal][open_count]
                        new_dp[bal-1][open_count] %= MOD
        dp = new_dp
    return dp[0][0]

s = input().strip()
print(solve(s))
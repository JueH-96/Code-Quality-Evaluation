MOD = 998244353

def count_non_ddos_strings(S):
    n = len(S)
    # dp[i][j] represents the number of ways to replace ?s in the prefix of length i
    # such that the last j characters can potentially form a DDoS-type string
    dp = [[0] * 4 for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        if S[i - 1] == '?':
            # If current character is '?', it can be replaced by any of 52 letters
            # Update dp for all possible lengths of potential DDoS-type string
            dp[i][0] = dp[i - 1][0] * 52 % MOD
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][1] * 52) % MOD
            dp[i][2] = (dp[i - 1][1] + dp[i - 1][2] * 52) % MOD
            dp[i][3] = (dp[i - 1][2] + dp[i - 1][3] * 52) % MOD
        else:
            # If current character is not '?', update dp based on whether it's uppercase or lowercase
            is_upper = S[i - 1].isupper()
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = dp[i - 1][1] if is_upper else dp[i - 1][0]
            dp[i][2] = dp[i - 1][2] if is_upper else (dp[i - 1][1] + dp[i - 1][2]) % MOD
            dp[i][3] = (dp[i - 1][3] if is_upper else 0) + dp[i - 1][2] * (1 if is_upper else 0)

    # The answer is the total number of ways to replace ?s minus the number of ways
    # that result in a DDoS-type string
    total_ways = pow(52, S.count('?'), MOD)
    return (total_ways - dp[n][3] + MOD) % MOD

# Read input
S = input().strip()

# Calculate and print the answer
print(count_non_ddos_strings(S))
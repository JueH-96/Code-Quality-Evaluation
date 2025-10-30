import sys

MOD = 998244353

def count_non_ddos_strings(S):
    n = len(S)
    q = S.count('?')

    # DP table to store the number of ways to form strings without DDoS-type subsequences
    dp = [[0] * (q + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(q + 1):
            if S[i - 1] == '?':
                # If the current character is '?', we can replace it with any letter
                dp[i][j + 1] = (dp[i][j + 1] + 52 * dp[i - 1][j]) % MOD
            else:
                # If the current character is not '?', we just copy the value from the previous state
                dp[i][j] = (dp[i][j] + dp[i - 1][j]) % MOD

    # Calculate the number of valid strings
    total_strings = pow(52, q, MOD)
    invalid_strings = sum(dp[n]) % MOD
    valid_strings = (total_strings - invalid_strings + MOD) % MOD

    return valid_strings

# Read input
S = sys.stdin.read().strip()

# Calculate and print the result
result = count_non_ddos_strings(S)
print(result)
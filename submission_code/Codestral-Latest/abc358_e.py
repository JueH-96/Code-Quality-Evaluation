MOD = 998244353

def count_valid_strings(K, C):
    # Initialize a DP table where dp[i][j] represents the number of valid strings of length i
    # with the given constraints on character counts.
    dp = [[0] * 27 for _ in range(K + 1)]
    dp[0][0] = 1  # Base case: one empty string

    for length in range(1, K + 1):
        for count in range(27):
            for i in range(27):
                if count + i <= 26 and dp[length - 1][count] > 0:
                    dp[length][count + i] += dp[length - 1][count]
                    dp[length][count + i] %= MOD

    total_count = 0
    for length in range(1, K + 1):
        for count in range(27):
            if all(dp[length][count] <= C[i] for i in range(26)):
                total_count += dp[length][count]
                total_count %= MOD

    return total_count

# Read input
K = int(input().strip())
C = list(map(int, input().strip().split()))

# Calculate and print the result
result = count_valid_strings(K, C)
print(result)
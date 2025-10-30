MOD = 998244353

def count_arithmetic_subsequences(A):
    N = len(A)
    dp = [[0] * (N + 1) for _ in range(N + 1)]

    for i in range(N):
        dp[i][1] = 1

    for length in range(2, N + 1):
        for i in range(N - length + 1):
            if 2 * A[i + 1] == A[i] + A[i + length - 1]:
                dp[i][length] = (dp[i + 1][length - 1] + dp[i][length - 1]) % MOD
            else:
                dp[i][length] = dp[i + 1][length - 1]

    result = [0] * N
    for length in range(1, N + 1):
        result[length - 1] = dp[0][length]

    return result

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the result
result = count_arithmetic_subsequences(A)

# Print the result
print(" ".join(map(str, result)))
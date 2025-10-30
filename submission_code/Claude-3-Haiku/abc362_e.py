from collections import defaultdict

def count_arithmetic_subsequences(A):
    N = len(A)
    mod = 998244353

    # Count the number of arithmetic subsequences of each length
    dp = [defaultdict(int) for _ in range(N+1)]
    for i in range(N):
        for j in range(i):
            diff = A[i] - A[j]
            dp[i+1][diff] = (dp[i+1][diff] + dp[i][diff] + 1) % mod

    # Print the answers
    result = []
    for i in range(1, N+1):
        total = sum(dp[i].values())
        result.append(total % mod)
    return ' '.join(map(str, result))

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve the problem
print(count_arithmetic_subsequences(A))
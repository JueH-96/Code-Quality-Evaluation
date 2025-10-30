import sys
input = sys.stdin.read
MOD = 998244353

def count_operations(N, A):
    # Initial grid configuration based on i % 2
    initial = [i % 2 for i in range(1, N + 1)]

    # Function to count the number of ways to transform the grid
    def count_ways(initial, target):
        n = len(initial)
        dp = [0] * (n + 1)
        dp[0] = 1

        for i in range(n):
            if initial[i] == target[i]:
                dp[i + 1] = dp[i] * 2 % MOD
            else:
                dp[i + 1] = dp[i]

        return dp[n]

    # Count the number of ways to achieve the target configuration
    result = count_ways(initial, A)
    print(result)

# Read input
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Solve the problem
count_operations(N, A)
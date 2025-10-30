MOD = 998244353

def count_operations(N, A):
    # Initialize the dp array to store the number of ways to achieve the desired configuration
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to have an empty configuration

    # Iterate through each cell
    for i in range(1, N + 1):
        if A[i - 1] == 1:
            # If A[i] is 1, we need to find the last 0 before this 1
            last_zero = i - 1
            while last_zero > 0 and A[last_zero - 1] == 1:
                last_zero -= 1
            if last_zero > 0:
                dp[i] = dp[last_zero - 1]
            else:
                dp[i] = 1
        else:
            dp[i] = dp[i - 1]

    return dp[N]

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the result
result = count_operations(N, A)

# Print the result
print(result)
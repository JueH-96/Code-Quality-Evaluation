MOD = 998244353

def count_permutations(N, A):
    # Initialize a list to store the number of valid permutations ending at each position
    dp = [0] * N
    dp[0] = 1  # There is one way to arrange the first element

    # Iterate through each element in the sequence
    for i in range(1, N):
        # If A_i is 0, it means P_i must be the largest in the current segment
        if A[i] == 0:
            dp[i] = dp[i - 1]
        else:
            # Otherwise, we need to consider the segment from A_i to i
            # The number of ways to arrange this segment is the sum of the ways to arrange the segments before A_i
            dp[i] = (dp[i - 1] + dp[A[i] - 1]) % MOD

    return dp[N - 1]

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the number of permutations
result = count_permutations(N, A)

# Print the result
print(result)
def count_sequences(N, A):
    MOD = 998244353

    # Precompute the number of segments of 1s and 0s
    segments = []
    current_value = A[0]
    current_length = 1

    for i in range(1, N):
        if A[i] == current_value:
            current_length += 1
        else:
            segments.append((current_value, current_length))
            current_value = A[i]
            current_length = 1
    segments.append((current_value, current_length))

    # Count the number of ways to merge segments
    num_segments = len(segments)
    if num_segments == 1:
        return 1  # Only one segment, no operations needed

    # Dynamic programming to count the ways
    dp = [0] * (num_segments + 1)
    dp[0] = 1  # Base case: one way to do nothing

    for i in range(1, num_segments + 1):
        dp[i] = dp[i - 1]  # Carry forward the previous count
        if segments[i - 1][0] == 1:  # If the current segment is 1
            dp[i] += dp[i - 1]  # We can merge with the previous segment
            dp[i] %= MOD

    return dp[num_segments]

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the result
result = count_sequences(N, A)

# Output the result
print(result)
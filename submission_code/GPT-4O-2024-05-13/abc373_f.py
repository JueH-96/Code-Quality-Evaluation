# YOUR CODE HERE
def max_happiness(N, W, items):
    # Initialize the dp array with -inf, except dp[0] which is 0
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0

    for w, v in items:
        # We need to process each item type independently
        new_dp = dp[:]
        for k in range(1, W // w + 1):
            happiness = k * v - k * k
            weight = k * w
            for j in range(W - weight + 1):
                if dp[j] != -float('inf'):
                    new_dp[j + weight] = max(new_dp[j + weight], dp[j] + happiness)
        dp = new_dp

    return max(dp)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
W = int(data[1])
items = [(int(data[i * 2 + 2]), int(data[i * 2 + 3])) for i in range(N)]

# Calculate and print the result
print(max_happiness(N, W, items))
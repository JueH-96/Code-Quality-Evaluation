import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
W = int(data[1])

weights = []
values = []

for i in range(n):
    weights.append(int(data[2 + 2 * i]))
    values.append(int(data[3 + 2 * i]))

# Define a function to calculate happiness
def happiness(k, v):
    return k * v - k * k

# Initialize the DP table
dp = [[0] * (W + 1) for _ in range(n + 1)]

# Fill the DP table
for i in range(1, n + 1):
    w = weights[i - 1]
    v = values[i - 1]
    for j in range(W + 1):
        dp[i][j] = dp[i - 1][j]
        if j >= w:
            for k in range(1, 10**10 + 1):
                if k * w > j:
                    break
                dp[i][j] = max(dp[i][j], dp[i - 1][j - k * w] + happiness(k, v))

# The answer is the maximum value in the last row of the DP table
print(max(dp[n]))
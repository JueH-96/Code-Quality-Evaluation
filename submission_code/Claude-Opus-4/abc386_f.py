# YOUR CODE HERE
K = int(input())
S = input().strip()
T = input().strip()

# Calculate edit distance using dynamic programming
m, n = len(S), len(T)

# dp[i][j] = minimum operations to convert S[0:i] to T[0:j]
dp = [[0] * (n + 1) for _ in range(m + 1)]

# Base cases
for i in range(m + 1):
    dp[i][0] = i  # Delete all characters from S
for j in range(n + 1):
    dp[0][j] = j  # Insert all characters to match T

# Fill the dp table
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if S[i-1] == T[j-1]:
            # Characters match, no operation needed
            dp[i][j] = dp[i-1][j-1]
        else:
            # Try all three operations and take minimum
            dp[i][j] = 1 + min(
                dp[i-1][j],    # Delete from S
                dp[i][j-1],    # Insert into S
                dp[i-1][j-1]   # Replace in S
            )

# Check if edit distance is at most K
if dp[m][n] <= K:
    print("Yes")
else:
    print("No")
import sys
sys.setrecursionlimit(1000000)  # Not necessary, but safe for potential future extensions

# Read the input string
S = input().strip()
N = len(S)

# Initialize a 2D DP table
dp = [[False for _ in range(N)] for _ in range(N)]

# All substrings of length 1 are palindromes
for i in range(N):
    dp[i][i] = True

# Variable to keep track of the maximum length
max_length = 1

# Check for substrings of length 2
for i in range(N - 1):
    if S[i] == S[i + 1]:
        dp[i][i + 1] = True
        max_length = max(max_length, 2)

# Check for substrings of length 3 to N
for length in range(3, N + 1):
    for i in range(N - length + 1):
        j = i + length - 1
        if S[i] == S[j] and dp[i + 1][j - 1]:
            dp[i][j] = True
            max_length = max(max_length, length)

# Output the maximum length
print(max_length)
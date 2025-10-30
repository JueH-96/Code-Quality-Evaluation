def edit_distance(s, t):
    # Using dynamic programming to find the minimum edit distance
    # dp[i][j] will hold the minimum edit distance between s[0:i] and t[0:j]
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the dp array
    for i in range(m + 1):
        dp[i][0] = i  # Cost of deleting all characters from s to match empty t
    for j in range(n + 1):
        dp[0][j] = j  # Cost of inserting all characters of t to match empty s
    
    # Fill the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Characters match, no operation needed
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + 1  # Substitution
                )
    
    return dp[m][n]

import sys
input = sys.stdin.read
data = input().split()
K = int(data[0])
S = data[1]
T = data[2]

# Calculate the minimum edit distance
min_operations = edit_distance(S, T)

# Output Yes if the minimum number of operations needed is <= K, otherwise No
if min_operations <= K:
    print("Yes")
else:
    print("No")
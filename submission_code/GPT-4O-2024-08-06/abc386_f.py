# YOUR CODE HERE
def can_transform(S, T, K):
    len_s = len(S)
    len_t = len(T)
    
    # Initialize the DP table
    dp = [[0] * (len_t + 1) for _ in range(len_s + 1)]
    
    # Base cases
    for i in range(len_s + 1):
        dp[i][0] = i  # Deleting all characters of S[0:i]
    for j in range(len_t + 1):
        dp[0][j] = j  # Inserting all characters of T[0:j]
    
    # Fill the DP table
    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            if S[i-1] == T[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,  # Delete
                    dp[i][j-1] + 1,  # Insert
                    dp[i-1][j-1] + 1 # Replace
                )
    
    # Check if the transformation is possible within K operations
    return dp[len_s][len_t] <= K

import sys
input = sys.stdin.read
data = input().split()
K = int(data[0])
S = data[1]
T = data[2]

if can_transform(S, T, K):
    print("Yes")
else:
    print("No")
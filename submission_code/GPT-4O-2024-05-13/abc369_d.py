# YOUR CODE HERE
def max_experience(N, A):
    # dp[i][0] is the max experience if we don't defeat the i-th monster
    # dp[i][1] is the max experience if we defeat the i-th monster
    dp = [[0, 0] for _ in range(N)]
    
    # Initial condition
    dp[0][0] = 0
    dp[0][1] = A[0]
    
    for i in range(1, N):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        dp[i][1] = dp[i-1][0] + A[i]
        if i >= 2:
            dp[i][1] = max(dp[i][1], dp[i-2][1] + 2 * A[i])
    
    return max(dp[N-1][0], dp[N-1][1])

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(max_experience(N, A))
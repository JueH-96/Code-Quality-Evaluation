# YOUR CODE HERE

import sys

def max_sum_subarray(arr):
    n = len(arr)
    dp = [[0, 0] for _ in range(n)]
    dp[0][0] = arr[0] if arr[0] % 2 == 0 else -sys.maxsize
    dp[0][1] = arr[0] if arr[0] % 2 == 1 else -sys.maxsize
    for i in range(1, n):
        if arr[i] % 2 == 0:
            dp[i][0] = max(dp[i-1][0] + arr[i], arr[i])
            dp[i][1] = dp[i-1][1]
        else:
            dp[i][0] = dp[i-1][0]
            dp[i][1] = max(dp[i-1][1] + arr[i], arr[i])
    return max(max(dp))

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    arr = list(map(int, input().strip().split()))
    print(max_sum_subarray(arr))
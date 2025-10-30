# YOUR CODE HERE
import sys

def max_alternating_subarray_sum(arr):
    n = len(arr)
    dp = [[0, 0] for _ in range(n)]
    dp[0] = [arr[0], 0]
    max_sum = arr[0]
    
    for i in range(1, n):
        if arr[i] % 2 == 0:
            dp[i][0] = max(arr[i], dp[i-1][1] + arr[i])
            dp[i][1] = dp[i-1][1]
        else:
            dp[i][1] = max(arr[i], dp[i-1][0] + arr[i])
            dp[i][0] = dp[i-1][0]
        max_sum = max(max_sum, dp[i][0], dp[i][1])
    
    return max_sum

t = int(input())
for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    print(max_alternating_subarray_sum(arr))
# YOUR CODE HERE
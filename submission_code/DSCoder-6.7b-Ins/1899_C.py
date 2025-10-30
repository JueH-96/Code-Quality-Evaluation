def max_subarray_sum(arr):
    n = len(arr)
    dp = [0]*n
    dp[0] = arr[0]
    for i in range(1, n):
        dp[i] = max(arr[i], dp[i-1]+arr[i] if (arr[i]%2!=arr[i-1]%2) else arr[i])
        dp[i] = max(dp[i], arr[i])
    return max(dp)

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    arr = list(map(int, input().strip().split()))
    print(max_subarray_sum(arr))
from collections import Counter

def max_pyramid_size(n, arr):
    count = Counter(arr)
    keys = sorted(count.keys())
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = count[keys[0]]
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + count[keys[i - 1]])
    return dp[-1]

n = int(input().strip())
arr = list(map(int, input().strip().split()))
print(max_pyramid_size(n, arr))
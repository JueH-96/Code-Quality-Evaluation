def solve():
    n = int(input())
    a = list(map(int, input().split()))
    if n == 0:
        print(0)
        return
    
    dp = [0] * n
    max_sum_overall = -float('inf')
    
    for i in range(n):
        dp[i] = a[i]
        if i > 0:
            if (a[i-1] % 2) != (a[i] % 2):
                dp[i] = max(dp[i], dp[i-1] + a[i])
        max_sum_overall = max(max_sum_overall, dp[i])
        
    print(max_sum_overall)

t = int(input())
for _ in range(t):
    solve()
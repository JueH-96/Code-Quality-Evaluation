def solve():
    n = int(input())
    a = list(map(int, input().split()))
    if n == 0:
        print(0)
        return
    
    dp = [0] * n
    dp[0] = a[0]
    max_so_far = dp[0]
    
    for i in range(1, n):
        if a[i-1] % 2 != a[i] % 2:
            dp[i] = max(a[i], dp[i-1] + a[i])
        else:
            dp[i] = a[i]
        max_so_far = max(max_so_far, dp[i])
        
    print(max_so_far)

t = int(input())
for _ in range(t):
    solve()
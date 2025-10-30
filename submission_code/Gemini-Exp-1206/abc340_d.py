def solve():
    n = int(input())
    stages = []
    for _ in range(n - 1):
        a, b, x = map(int, input().split())
        stages.append((a, b, x))

    dp = [float('inf')] * (n + 1)
    dp[1] = 0
    
    for i in range(1, n):
        if dp[i] == float('inf'):
            continue
        
        a, b, x = stages[i-1]
        
        dp[i+1] = min(dp[i+1], dp[i] + a)
        dp[x] = min(dp[x], dp[i] + b)

    print(dp[n])

solve()
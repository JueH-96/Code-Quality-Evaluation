def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    
    mod = 998244353

    dp = [[0] * (x + 1) for _ in range(x + 2)]
    dp[0][0] = 1
    
    for time in range(x + 1):
        for prev_time in range(x + 1):
            if dp[prev_time][time] == 0:
                continue
            
            for i in range(n):
                next_time = time + t[i]
                if next_time <= x:
                    dp[next_time][time] = (dp[next_time][time] + dp[prev_time][time] * pow(n, mod - 2, mod)) % mod
                elif next_time == x + 1:
                    dp[next_time][time] = (dp[next_time][time] + dp[prev_time][time] * pow(n, mod - 2, mod)) % mod

    ans = 0
    for prev_time in range(x + 1):
        if dp[x + 1][prev_time] == 0:
            continue
        
        for i in range(n):
            if i == 0:
                if prev_time + t[i] > x:
                    ans = (ans + dp[x + 1][prev_time] * pow(n, mod - 2, mod)) % mod
    
    
    print(ans)

solve()
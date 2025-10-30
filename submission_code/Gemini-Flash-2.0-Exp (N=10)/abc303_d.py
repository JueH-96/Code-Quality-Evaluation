def solve():
    x, y, z = map(int, input().split())
    s = input()
    
    n = len(s)
    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    dp[0][0] = 0
    
    for i in range(n):
        for caps_lock in range(2):
            if dp[i][caps_lock] == float('inf'):
                continue
            
            # Press 'a' key
            if (caps_lock == 0 and s[i] == 'a') or (caps_lock == 1 and s[i] == 'A'):
                dp[i+1][caps_lock] = min(dp[i+1][caps_lock], dp[i][caps_lock] + x)
            
            # Press 'a' key with shift
            if (caps_lock == 0 and s[i] == 'A') or (caps_lock == 1 and s[i] == 'a'):
                dp[i+1][caps_lock] = min(dp[i+1][caps_lock], dp[i][caps_lock] + y)
            
            # Press Caps Lock
            dp[i][1-caps_lock] = min(dp[i][1-caps_lock], dp[i][caps_lock] + z)
            
    print(min(dp[n][0], dp[n][1]))

solve()
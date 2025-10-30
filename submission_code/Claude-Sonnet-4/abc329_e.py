def solve():
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()
    
    # dp[i] = True if we can construct S[0:i] using the operations
    dp = [False] * (N + 1)
    dp[0] = True
    
    for i in range(N + 1):
        if not dp[i]:
            continue
            
        # Try placing T starting at position i
        if i + M <= N and S[i:i + M] == T:
            dp[i + M] = True
    
    print("Yes" if dp[N] else "No")

solve()
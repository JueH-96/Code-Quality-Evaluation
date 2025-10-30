def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    X, Y, Z = map(int, data[:3])
    S = data[3]
    
    n = len(S)
    
    # dp[i][c] = minimum cost to have typed first i characters
    #            with CapsLock state c (0=off, 1=on)
    # We'll build S[0..i-1] (i characters) using the minimal cost.
    # We start with caps lock off: dp[0][0] = 0
    # If we want to start with caps lock on, we can press it once: dp[0][1] = Z
    # Otherwise dp[0][1] = +∞ initially.
    
    INF = float('inf')
    dp = [[INF, INF] for _ in range(n+1)]
    
    dp[0][0] = 0
    dp[0][1] = Z  # We can toggle caps at the very beginning if needed
    
    def cost_press(ch, caps):
        # Return cost to type character ch when caps state is caps
        # 'caps' = 0 means CapsLock is off, 1 means on.
        if ch == 'a':
            return X if caps == 0 else Y
        else:  # ch == 'A'
            return X if caps == 1 else Y
    
    for i in range(n):
        for c in (0, 1):
            if dp[i][c] == INF:
                continue
            # Option 1: Type the next character with the current caps state c
            dp[i+1][c] = min(dp[i+1][c], dp[i][c] + cost_press(S[i], c))
            
            # Option 2: Toggle caps, then type
            nc = 1 - c  # new caps state
            dp[i+1][nc] = min(dp[i+1][nc], dp[i][c] + Z + cost_press(S[i], nc))
    
    # We can end with either caps=0 or caps=1
    ans = min(dp[n][0], dp[n][1])
    print(ans)

# Call main() at the end
if __name__ == "__main__":
    main()
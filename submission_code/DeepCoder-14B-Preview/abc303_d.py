def main():
    import sys
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    n = len(S)
    INF = float('inf')
    
    # Initialize DP table
    dp = [ [INF] * 2 for _ in range(n+1) ]
    dp[0][0] = 0  # Starting with Caps Lock off
    
    for i in range(n):
        for c_prev in [0, 1]:
            if dp[i][c_prev] == INF:
                continue
            
            # Press action3 (toggle Caps Lock)
            new_state = 1 - c_prev
            new_cost = dp[i][c_prev] + Z
            if new_cost < dp[i][new_state]:
                dp[i][new_state] = new_cost
            
            # Press action1 or action2
            for c in [0, 1]:
                cost_toggle = 0 if (c == c_prev) else Z
                
                # Action1: 'a' if c is 0, else 'A'
                char = 'a' if c == 0 else 'A'
                if S[i] == char:
                    total_cost = dp[i][c_prev] + cost_toggle + X
                    if total_cost < dp[i+1][c]:
                        dp[i+1][c] = total_cost
                
                # Action2: 'A' if c is 0, else 'a'
                char = 'A' if c == 0 else 'a'
                if S[i] == char:
                    total_cost = dp[i][c_prev] + cost_toggle + Y
                    if total_cost < dp[i+1][c]:
                        dp[i+1][c] = total_cost
    
    # The minimal cost is the minimum of the two possible states after typing all characters
    print(min(dp[n][0], dp[n][1]))

if __name__ == "__main__":
    main()
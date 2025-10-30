def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Aoki's moves: R, P, S
    # Takahashi's winning moves: P, S, R (respectively)
    # Takahashi's moves must alternate
    
    # Map Aoki's move to Takahashi's winning move
    win_move = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # We need to find the maximum number of games Takahashi could have won
    # while ensuring his moves alternate.
    
    # We will use dynamic programming to solve this.
    # dp[i][0] - max wins using move 'R' at position i
    # dp[i][1] - max wins using move 'P' at position i
    # dp[i][2] - max wins using move 'S' at position i
    
    # Initialize dp table
    dp = [[0] * 3 for _ in range(N + 1)]
    
    # Fill the dp table
    for i in range(1, N + 1):
        aoki_move = S[i - 1]
        takahashi_win = win_move[aoki_move]
        
        # If Takahashi plays 'R'
        dp[i][0] = max(dp[i - 1][1], dp[i - 1][2]) + (1 if takahashi_win == 'R' else 0)
        # If Takahashi plays 'P'
        dp[i][1] = max(dp[i - 1][0], dp[i - 1][2]) + (1 if takahashi_win == 'P' else 0)
        # If Takahashi plays 'S'
        dp[i][2] = max(dp[i - 1][0], dp[i - 1][1]) + (1 if takahashi_win == 'S' else 0)
    
    # The answer is the maximum of the last row of dp table
    result = max(dp[N][0], dp[N][1], dp[N][2])
    
    print(result)

if __name__ == "__main__":
    main()
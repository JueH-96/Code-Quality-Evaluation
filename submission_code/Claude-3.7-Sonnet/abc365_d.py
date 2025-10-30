def max_wins(N, S):
    # dp[i][j] is the maximum number of games Takahashi can win up to and including the i-th game,
    # if his i-th move is j (0 for R, 1 for P, 2 for S).
    dp = [[-float('inf') for _ in range(3)] for _ in range(N)]
    
    # Rock, Paper, Scissors symbols
    R, P, SC = 0, 1, 2
    
    # For the first game
    if S[0] == 'R':
        dp[0][R] = 0  # Takahashi plays R (draw)
        dp[0][P] = 1  # Takahashi plays P (win)
    elif S[0] == 'P':
        dp[0][P] = 0  # Takahashi plays P (draw)
        dp[0][SC] = 1  # Takahashi plays S (win)
    else:  # S[0] == 'S'
        dp[0][SC] = 0  # Takahashi plays S (draw)
        dp[0][R] = 1  # Takahashi plays R (win)
    
    # For subsequent games
    for i in range(1, N):
        if S[i] == 'R':
            # Takahashi can play R (draw) or P (win)
            if dp[i-1][P] != -float('inf'):
                dp[i][R] = dp[i-1][P]
            if dp[i-1][SC] != -float('inf'):
                dp[i][R] = max(dp[i][R], dp[i-1][SC])
            
            if dp[i-1][R] != -float('inf'):
                dp[i][P] = dp[i-1][R] + 1
            if dp[i-1][SC] != -float('inf'):
                dp[i][P] = max(dp[i][P], dp[i-1][SC] + 1)
        
        elif S[i] == 'P':
            # Takahashi can play P (draw) or S (win)
            if dp[i-1][R] != -float('inf'):
                dp[i][P] = dp[i-1][R]
            if dp[i-1][SC] != -float('inf'):
                dp[i][P] = max(dp[i][P], dp[i-1][SC])
            
            if dp[i-1][R] != -float('inf'):
                dp[i][SC] = dp[i-1][R] + 1
            if dp[i-1][P] != -float('inf'):
                dp[i][SC] = max(dp[i][SC], dp[i-1][P] + 1)
        
        else:  # S[i] == 'S'
            # Takahashi can play S (draw) or R (win)
            if dp[i-1][R] != -float('inf'):
                dp[i][SC] = dp[i-1][R]
            if dp[i-1][P] != -float('inf'):
                dp[i][SC] = max(dp[i][SC], dp[i-1][P])
            
            if dp[i-1][P] != -float('inf'):
                dp[i][R] = dp[i-1][P] + 1
            if dp[i-1][SC] != -float('inf'):
                dp[i][R] = max(dp[i][R], dp[i-1][SC] + 1)
    
    result = -float('inf')
    for j in range(3):
        if dp[N-1][j] != -float('inf'):
            result = max(result, dp[N-1][j])
    
    return result

# Read input
N = int(input())
S = input()

# Print the answer
print(max_wins(N, S))
def get_winning_move(opponent_move):
    if opponent_move == 'R':
        return 'P'
    elif opponent_move == 'P':
        return 'S'
    else:
        return 'R'

def get_draw_move(opponent_move):
    return opponent_move

N = int(input())
S = input()

# dp[i][j] represents max wins up to index i when using move j
# j: 0=R, 1=P, 2=S
dp = [[-1]*3 for _ in range(N)]

# Initialize first position
for j in range(3):
    moves = 'RPS'
    my_move = moves[j]
    opp_move = S[0]
    if my_move == get_winning_move(opp_move):
        dp[0][j] = 1
    elif my_move == get_draw_move(opp_move):
        dp[0][j] = 0
    else:
        dp[0][j] = -float('inf')  # Invalid as we can't lose

# Fill dp table
for i in range(1, N):
    for j in range(3):
        moves = 'RPS'
        my_move = moves[j]
        opp_move = S[i]
        
        # Check if current move wins/draws against opponent
        curr_result = 0
        if my_move == get_winning_move(opp_move):
            curr_result = 1
        elif my_move == get_draw_move(opp_move):
            curr_result = 0
        else:
            continue  # Skip if we would lose
            
        # Try all previous moves except current one (must be different)
        max_prev = -float('inf')
        for prev_j in range(3):
            if prev_j != j and dp[i-1][prev_j] != -float('inf'):
                max_prev = max(max_prev, dp[i-1][prev_j])
                
        if max_prev != -float('inf'):
            dp[i][j] = max_prev + curr_result

# Find maximum wins possible
result = 0
for j in range(3):
    if dp[N-1][j] != -float('inf'):
        result = max(result, dp[N-1][j])

print(result)
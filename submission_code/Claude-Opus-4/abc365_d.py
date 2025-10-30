# YOUR CODE HERE
def solve():
    N = int(input())
    S = input().strip()
    
    # Define moves
    ROCK, PAPER, SCISSORS = 0, 1, 2
    
    # Convert string to move indices
    def char_to_move(c):
        if c == 'R': return ROCK
        elif c == 'P': return PAPER
        else: return SCISSORS
    
    # Get winning move against opponent's move
    def get_winning_move(opponent_move):
        if opponent_move == ROCK: return PAPER
        elif opponent_move == PAPER: return SCISSORS
        else: return ROCK
    
    # Check if move1 wins against move2
    def wins(move1, move2):
        return (move1 == PAPER and move2 == ROCK) or \
               (move1 == SCISSORS and move2 == PAPER) or \
               (move1 == ROCK and move2 == SCISSORS)
    
    # Convert Aoki's moves
    aoki_moves = [char_to_move(c) for c in S]
    
    # dp[i][j] = maximum wins from game i to end, if Takahashi played move j in game i-1
    # j can be 0 (ROCK), 1 (PAPER), 2 (SCISSORS), or 3 (no previous move for i=0)
    dp = [[-1] * 4 for _ in range(N + 1)]
    
    # Base case: no games left
    for j in range(4):
        dp[N][j] = 0
    
    # Fill dp table backwards
    for i in range(N - 1, -1, -1):
        for prev_move in range(4):
            max_wins = 0
            
            # Try each possible move for Takahashi
            for curr_move in range(3):
                # Check if this move is valid
                if prev_move < 3 and curr_move == prev_move:
                    continue  # Can't repeat the same move
                
                # Check if Takahashi doesn't lose
                if wins(aoki_moves[i], curr_move):
                    continue  # Aoki wins, invalid
                
                # Calculate wins from this position
                wins_here = 1 if wins(curr_move, aoki_moves[i]) else 0
                total_wins = wins_here + dp[i + 1][curr_move]
                max_wins = max(max_wins, total_wins)
            
            dp[i][prev_move] = max_wins
    
    # The answer is the maximum wins starting from game 0 with no previous move
    print(dp[0][3])

solve()
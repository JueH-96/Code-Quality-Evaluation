def main():
    N = int(input())
    S = input().strip()
    
    # Define the winning moves
    win = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Initialize the maximum wins
    max_wins = 0
    
    # We need to try all possible starting moves for Takahashi
    # Since the first move can be any of R, P, S, we try all three
    # and choose the one that maximizes the wins
    
    # Initialize the best result
    best = 0
    
    # Try starting with R
    prev = 'R'
    wins = 0
    for i in range(N):
        # Determine the move that wins or ties
        if prev == win[S[i]]:
            wins += 1
        # Next move must be different
        if i < N-1:
            # Choose the next move that is different from the current
            # and maximizes the wins
            # We have two choices: the other two moves
            # We choose the one that wins the next game if possible
            # Otherwise, choose the one that ties
            # So, for the next move, we have two options:
            # option1: the move that wins S[i+1]
            # option2: the move that ties S[i+1]
            # We choose the one that maximizes the wins
            # So, we need to look ahead
            # But since we can't look ahead, we need to choose the move that is different
            # and can win or tie
            # So, for the next move, we have two options:
            # option1: the move that is different from prev and wins S[i+1]
            # option2: the move that is different from prev and ties S[i+1]
            # We choose the one that wins if possible
            # So, we need to find the move that is different from prev and wins S[i+1]
            # If such a move exists, choose it
            # Otherwise, choose the move that is different from prev and ties S[i+1]
            # So, we need to find the move that is different from prev and is win[S[i+1]]
            # If it is different from prev, choose it
            # Otherwise, choose the move that is different from prev and is S[i+1]
            # So, we have:
            # next_move = win[S[i+1]] if win[S[i+1]] != prev else S[i+1]
            # But wait, if win[S[i+1]] == prev, then S[i+1] must be the move that loses to prev
            # So, the move that ties is S[i+1]
            # So, next_move = win[S[i+1]] if win[S[i+1]] != prev else S[i+1]
            next_move = win[S[i+1]] if win[S[i+1]] != prev else S[i+1]
            prev = next_move
    best = max(best, wins)
    
    # Try starting with P
    prev = 'P'
    wins = 0
    for i in range(N):
        if prev == win[S[i]]:
            wins += 1
        if i < N-1:
            next_move = win[S[i+1]] if win[S[i+1]] != prev else S[i+1]
            prev = next_move
    best = max(best, wins)
    
    # Try starting with S
    prev = 'S'
    wins = 0
    for i in range(N):
        if prev == win[S[i]]:
            wins += 1
        if i < N-1:
            next_move = win[S[i+1]] if win[S[i+1]] != prev else S[i+1]
            prev = next_move
    best = max(best, wins)
    
    print(best)

if __name__ == "__main__":
    main()
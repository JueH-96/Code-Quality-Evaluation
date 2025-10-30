def find_max_wins(N, S):
    # Mapping of what beats what
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Initialize the previous move and the number of wins
    prev_move = ''
    wins = 0
    
    for move in S:
        # Determine the winning move against Aoki's move
        winning_move = beats[move]
        
        # If the winning move is different from the previous move, Takahashi wins
        if winning_move != prev_move:
            wins += 1
            prev_move = winning_move
        else:
            # Otherwise, Takahashi has to switch to a move that doesn't lose
            # but also doesn't win, so he doesn't win this round
            # We just pick a move that isn't Aoki's move or the previous move
            for m in beats:
                if m != move and m != prev_move:
                    prev_move = m
                    break
    
    return wins

# Read input from stdin
N = int(input().strip())
S = input().strip()

# Calculate and print the result
print(find_max_wins(N, S))
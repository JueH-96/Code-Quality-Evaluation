# YOUR CODE HERE
def max_wins(N, S):
    # Mapping from Aoki's move to Takahashi's winning move
    win_move = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Initialize variables
    last_move = None
    win_count = 0
    
    # Iterate over each of Aoki's moves
    for i in range(N):
        # Determine the winning move for Takahashi
        current_move = win_move[S[i]]
        
        # Ensure Takahashi's move is different from the last move
        if current_move == last_move:
            # If it's the same, we need to choose a different move
            # Since there are only 3 moves, we can choose the remaining one
            # Find an alternative move that is not the same as the last move
            for move in win_move.values():
                if move != last_move and move != current_move:
                    current_move = move
                    break
        
        # Update the last move
        last_move = current_move
        
        # Takahashi wins this game
        win_count += 1
    
    return win_count

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
S = data[1]

# Calculate and print the result
result = max_wins(N, S)
print(result)
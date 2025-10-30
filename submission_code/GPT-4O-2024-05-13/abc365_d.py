# YOUR CODE HERE
def max_wins(N, S):
    # Mapping of what beats what
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Initialize the count of wins
    wins = 0
    
    # Initialize the previous move of Takahashi
    prev_move = ''
    
    for i in range(N):
        # Determine the move Takahashi should play to win
        current_move = beats[S[i]]
        
        # Ensure the move is different from the previous move
        if current_move == prev_move:
            # If it's the same, we need to change it to another valid move
            for move in ['R', 'P', 'S']:
                if move != prev_move and move != beats[S[i]]:
                    current_move = move
                    break
        
        # Update the previous move
        prev_move = current_move
        
        # Check if Takahashi wins this game
        if current_move == beats[S[i]]:
            wins += 1
    
    return wins

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Calculate and print the result
print(max_wins(N, S))
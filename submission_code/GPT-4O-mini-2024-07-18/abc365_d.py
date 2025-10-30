def max_wins(N, S):
    # Initialize the count of wins
    wins = 0
    # Initialize the last move of Takahashi
    last_move = None
    
    # Define a mapping of Aoki's moves to Takahashi's winning moves
    winning_moves = {
        'R': 'P',  # Takahashi plays Paper to beat Rock
        'P': 'S',  # Takahashi plays Scissors to beat Paper
        'S': 'R'   # Takahashi plays Rock to beat Scissors
    }
    
    for i in range(N):
        # Determine the move Takahashi should play to win against Aoki's move
        current_move = winning_moves[S[i]]
        
        # If the current move is the same as the last move, we need to switch
        if current_move == last_move:
            # Switch to the other winning move
            if current_move == 'P':
                current_move = 'R'  # Switch to Rock
            elif current_move == 'S':
                current_move = 'P'  # Switch to Paper
            else:  # current_move == 'R'
                current_move = 'S'  # Switch to Scissors
        
        # Count the win
        wins += 1
        # Update the last move
        last_move = current_move
    
    return wins

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Get the result and print it
result = max_wins(N, S)
print(result)
def determine_winner(N, A):
    # Calculate the total number of moves needed to fill the set S
    total_moves = sum(A)
    
    # The number of unique indices that need to be filled in S is N
    # The game ends when S contains all indices from 1 to N
    # Fennec starts first, so if the total moves are odd, Fennec wins, otherwise Snuke wins
    if total_moves % 2 == 1:
        return "Fennec"
    else:
        return "Snuke"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Determine the winner
winner = determine_winner(N, A)

# Print the result
print(winner)
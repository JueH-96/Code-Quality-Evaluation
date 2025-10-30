# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    
    S = []
    for _ in range(N):
        S.append(sys.stdin.readline().strip())
    
    player_scores = [] # Stores (wins, player_number)
    
    # Calculate wins for each player
    # Player numbers are 1-indexed, so player i corresponds to S[i-1] (0-indexed string)
    for i in range(N): # i is the 0-indexed string index, representing player i+1
        player_number = i + 1
        wins = 0
        # Iterate through the characters of S[i] to count 'o's
        for j in range(N):
            if S[i][j] == 'o':
                wins += 1
        player_scores.append((wins, player_number))
    
    # Sort players based on ranking criteria:
    # 1. More wins ranks higher (descending order of wins)
    # 2. Smaller player number ranks higher if wins are equal (ascending order of player number)
    # To achieve this with Python's default ascending sort:
    # - Negate wins for descending order
    # - Keep player_number as is for ascending order
    # So, the key for sorting will be (-wins, player_number)
    player_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract and print the sorted player numbers
    ranked_players = [player_data[1] for player_data in player_scores]
    
    # Print the result space-separated
    print(*ranked_players)

# Call the solve function to run the program
solve()
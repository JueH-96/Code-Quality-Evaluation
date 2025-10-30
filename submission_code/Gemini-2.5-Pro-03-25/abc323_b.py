# YOUR CODE HERE
import sys

# Read the number of players
n = int(sys.stdin.readline())

# Read the match results for each player
# results[i] contains the results string for player i+1
results = [sys.stdin.readline().strip() for _ in range(n)]

# Create a list to store player data (player number and win count)
players_data = []

# Iterate through each player (0 to N-1 corresponding to players 1 to N)
for i in range(n):
    # Player number is i + 1
    player_number = i + 1
    # Get the result string for this player
    result_string = results[i]
    # Count the number of wins ('o') for this player
    wins = result_string.count('o')
    # Store the player number and their win count
    players_data.append({'number': player_number, 'wins': wins})

# Sort the players based on the ranking criteria:
# 1. Primary key: Number of wins (descending). We use -wins for ascending sort based on the negative value.
# 2. Secondary key: Player number (ascending).
# The sorted function with a lambda key achieves this multi-level sorting.
sorted_players_data = sorted(players_data, key=lambda p: (-p['wins'], p['number']))

# Extract the player numbers from the sorted list
ranked_player_numbers = [p['number'] for p in sorted_players_data]

# Print the sorted player numbers, separated by spaces
# The * operator unpacks the list into individual arguments for the print function
print(*ranked_player_numbers)

# END OF YOUR CODE
# YOUR CODE HERE
N = int(input())
results = [input() for _ in range(N)]

# Count wins for each player
wins = [result.count('o') for result in results]

# Create a list of (player_number, wins) pairs
players_with_wins = [(i+1, wins[i]) for i in range(N)]

# Sort in descending order based on wins, and if there's a tie, sort in ascending order based on player number
players_with_wins.sort(key=lambda x: (-x[1], x[0]))

# Extract player numbers
ranked_players = [player for player, _ in players_with_wins]

print(' '.join(map(str, ranked_players)))
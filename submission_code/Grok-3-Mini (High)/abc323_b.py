import sys

# Read N from stdin
N = int(sys.stdin.readline().strip())

# Read N strings from stdin
strings = [sys.stdin.readline().strip() for _ in range(N)]

# Create a list of tuples (player_number, wins)
players = [(num, strings[num-1].count('o')) for num in range(1, N+1)]

# Sort the players: descending wins, ascending player number if tie
sorted_players = sorted(players, key=lambda x: (-x[1], x[0]))

# Extract player numbers and convert to strings
result = [str(p[0]) for p in sorted_players]

# Print the result separated by spaces
print(' '.join(result))
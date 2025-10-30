# YOUR CODE HERE
n = int(input())
results = []
for _ in range(n):
    results.append(input())

# Count wins for each player
player_wins = []
for i in range(n):
    wins = results[i].count('o')
    player_wins.append((wins, i + 1))

# Sort by wins (descending) and then by player number (ascending)
player_wins.sort(key=lambda x: (-x[0], x[1]))

# Extract player numbers and print
print(' '.join(str(x[1]) for x in player_wins))
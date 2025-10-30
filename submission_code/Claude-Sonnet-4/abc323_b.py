# YOUR CODE HERE
n = int(input())
results = []
for i in range(n):
    s = input().strip()
    results.append(s)

# Count wins for each player
players_wins = []
for i in range(n):
    wins = 0
    for j in range(n):
        if results[i][j] == 'o':
            wins += 1
    players_wins.append((wins, i + 1))  # (wins, player_number)

# Sort by wins descending, then by player number ascending
players_wins.sort(key=lambda x: (-x[0], x[1]))

# Output player numbers
output = []
for wins, player_num in players_wins:
    output.append(str(player_num))

print(' '.join(output))
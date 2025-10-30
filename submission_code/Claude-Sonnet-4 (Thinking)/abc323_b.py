n = int(input())
strings = []
for i in range(n):
    strings.append(input())

# Count wins for each player
wins = []
for i in range(n):
    win_count = 0
    for j in range(n):
        if strings[i][j] == 'o':
            win_count += 1
    wins.append((win_count, i + 1))  # (wins, player_number)

# Sort by wins (descending), then by player_number (ascending)
wins.sort(key=lambda x: (-x[0], x[1]))

# Output player numbers
result = [str(x[1]) for x in wins]
print(' '.join(result))
N = int(input())
S = [input() for _ in range(N)]

# Count wins for each player
wins = []
for i in range(N):
    win_count = S[i].count('o')
    wins.append((win_count, -i, i+1))  # Store (wins, -player_num, player_num)

# Sort by wins (descending) and player number (ascending when ties)
wins.sort(reverse=True)

# Print player numbers in order
print(*[x[2] for x in wins])
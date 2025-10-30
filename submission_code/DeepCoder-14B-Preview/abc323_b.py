n = int(input())
s = [input().strip() for _ in range(n)]

# Calculate the number of wins for each player
player_wins = [(i, s[i-1].count('o')) for i in range(1, n+1)]

# Sort the players based on the criteria
sorted_players = sorted(player_wins, key=lambda x: (-x[1], x[0]))

# Extract the player numbers in the sorted order
result = [str(p[0]) for p in sorted_players]

print(' '.join(result))
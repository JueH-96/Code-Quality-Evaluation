n = int(input())
strings = [input().strip() for _ in range(n)]

players = []
for i in range(1, n + 1):
    s = strings[i - 1]
    wins = 0
    for j in range(n):
        if j != i - 1 and s[j] == 'o':
            wins += 1
    players.append((i, wins))

# Sort by wins descending, then by player number ascending
players_sorted = sorted(players, key=lambda x: (-x[1], x[0]))

result = [str(p[0]) for p in players_sorted]
print(' '.join(result))
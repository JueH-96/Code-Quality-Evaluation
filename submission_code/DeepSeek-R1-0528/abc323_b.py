n = int(input().strip())
strings = [input().strip() for _ in range(n)]

wins = [s.count('o') for s in strings]

players = []
for i in range(n):
    players.append((wins[i], i + 1))

sorted_players = sorted(players, key=lambda x: (-x[0], x[1]))

result = [str(player[1]) for player in sorted_players]
print(' '.join(result))
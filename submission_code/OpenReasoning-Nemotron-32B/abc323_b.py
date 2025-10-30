n = int(input().strip())
strings = [input().strip() for _ in range(n)]

players = []
for i in range(n):
	s = strings[i]
	wins = s.count('o')
	players.append((wins, i + 1))

sorted_players = sorted(players, key=lambda x: (-x[0], x[1]))
result = [str(player[1]) for player in sorted_players]
print(" ".join(result))
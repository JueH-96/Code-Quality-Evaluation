n = int(input().strip())
players = []
for i in range(n):
	s = input().strip()
	wins = s.count('o')
	players.append((wins, i + 1))

players.sort(key=lambda x: (-x[0], x[1]))

result = [str(p[1]) for p in players]
print(" ".join(result))
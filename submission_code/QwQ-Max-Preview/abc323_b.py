n = int(input())
players = []
for i in range(n):
    s = input().strip()
    wins = s.count('o')
    players.append((-wins, i + 1))  # i+1 for 1-based player number

players.sort()
result = [str(p[1]) for p in players]
print(' '.join(result))
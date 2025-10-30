N = int(input())
S = [input().strip() for _ in range(N)]
players = []
for i in range(N):
    wins = S[i].count('o')
    players.append((-wins, i + 1))
players.sort()
result = [str(player[1]) for player in players]
print(' '.join(result))
n = int(input())
s_list = [input().strip() for _ in range(n)]
players = []
for i in range(1, n+1):
    s = s_list[i-1]
    wins = s.count('o')
    players.append((-wins, i))
players.sort()
result = [str(p[1]) for p in players]
print(' '.join(result))
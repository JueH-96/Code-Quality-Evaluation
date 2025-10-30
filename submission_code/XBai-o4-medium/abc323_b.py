n = int(input())
s_list = [input().strip() for _ in range(n)]
players = []
for i in range(n):
    win_count = s_list[i].count('o')
    players.append((-win_count, i + 1))
sorted_players = sorted(players)
result = [str(p[1]) for p in sorted_players]
print(' '.join(result))
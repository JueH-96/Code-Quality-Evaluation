n = int(input())
S = [input().strip() for _ in range(n)]

players = []
for i in range(1, n + 1):
    s = S[i-1]
    wins = 0
    for j in range(n):
        if j == i - 1:
            continue
        if s[j] == 'o':
            wins += 1
    players.append((-wins, i))

players.sort()
result = [str(p[1]) for p in players]

print(' '.join(result))
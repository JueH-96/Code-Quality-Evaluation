N = int(input())
S = [input() for _ in range(N)]

wins = [0] * N
for i in range(N):
    for j in range(N):
        if S[i][j] == 'o':
            wins[i] += 1

for i in [x[0] for x in sorted(enumerate(wins), key=lambda x: (-x[1], x[0]))]:
    print(i + 1)
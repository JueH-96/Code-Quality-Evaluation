# YOUR CODE HERE

N = int(input())
S = [input() for _ in range(N)]

wins = [0]*N

for i in range(N):
    for j in range(N):
        if i != j and S[i][j] == 'o':
            wins[i] += 1
        elif i == j and S[i][j] == 'x':
            wins[i] += 1

players = list(range(1, N+1))
players.sort(key=lambda x: (-wins[x-1], x))

for player in players:
    print(player)
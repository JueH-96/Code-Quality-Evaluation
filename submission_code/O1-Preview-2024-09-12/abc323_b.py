# YOUR CODE HERE
N = int(input())
S = [input() for _ in range(N)]

wins = [0]*N

for i in range(N):
    for j in range(N):
        if i != j and S[i][j] == 'o':
            wins[i] += 1

sorted_players = sorted(range(1, N+1), key=lambda x: (-wins[x-1], x))
print(' '.join(map(str, sorted_players)))
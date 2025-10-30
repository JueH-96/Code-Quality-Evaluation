# YOUR CODE HERE

N = int(input())
matches = [input() for _ in range(N)]

wins = [0]*N

for i in range(N):
    for j in range(N):
        if i != j and matches[i][j] == 'o':
            wins[i] += 1

rankings = sorted(range(1, N+1), key=lambda i: (-wins[i-1], i))

print(*rankings)
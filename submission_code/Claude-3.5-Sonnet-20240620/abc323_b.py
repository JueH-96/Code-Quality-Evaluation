# YOUR CODE HERE
N = int(input())
results = [input() for _ in range(N)]

wins = [0] * N
for i in range(N):
    wins[i] = results[i].count('o')

players = list(range(1, N+1))
players.sort(key=lambda x: (-wins[x-1], x))

print(*players)
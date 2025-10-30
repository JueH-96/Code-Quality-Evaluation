import sys

N = int(input())
S = [input().strip() for _ in range(N)]

# Calculate the number of wins for each player
wins = [0] * N
for i in range(N):
    for j in range(N):
        if S[i][j] == 'o':
            wins[i] += 1

# Sort players by wins (descending) and then by player number (ascending)
players = sorted(range(1, N + 1), key=lambda x: (-wins[x - 1], x))

# Print the sorted player numbers
print(" ".join(map(str, players)))
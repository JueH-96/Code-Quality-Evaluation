# YOUR CODE HERE
import sys

# Read input
N = int(sys.stdin.readline().strip())
S = [sys.stdin.readline().strip() for _ in range(N)]

# Initialize win counts
wins = [0] * N

# Count wins for each player
for i in range(N):
    for j in range(N):
        if S[i][j] == 'o':
            wins[i] += 1

# Sort players by wins and player number
sorted_players = sorted(range(N), key=lambda x: (-wins[x], x))

# Print sorted player numbers
print(' '.join(map(str, sorted_players + 1)))
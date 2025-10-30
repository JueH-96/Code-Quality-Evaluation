# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
results = data[1:]

# Calculate the number of wins for each player
wins = [0] * N
for i in range(N):
    for j in range(N):
        if results[i][j] == 'o':
            wins[i] += 1

# Create a list of tuples (wins, player_number) and sort it
ranking = sorted([(-wins[i], i) for i in range(N)])

# Print the player numbers in descending order of rank
print(' '.join(str(rank[1] + 1) for rank in ranking))
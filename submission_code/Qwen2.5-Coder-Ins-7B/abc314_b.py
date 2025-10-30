# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
bets = []
index = 1
for i in range(N):
    C = int(data[index])
    A = list(map(int, data[index + 1:index + 1 + C]))
    bets.append((A, C))
    index += 1 + C
X = int(data[index])

# Find people who bet on X with the fewest bets
min_bets = float('inf')
people = []
for i in range(N):
    if X in bets[i][0]:
        if bets[i][1] < min_bets:
            min_bets = bets[i][1]
            people = [i + 1]
        elif bets[i][1] == min_bets:
            people.append(i + 1)

# Print the result
print(len(people))
print(' '.join(map(str, people)))
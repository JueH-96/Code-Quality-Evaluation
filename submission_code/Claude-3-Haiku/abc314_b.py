import sys

# Read input
N = int(sys.stdin.readline().strip())
bets = []
for _ in range(N):
    C = int(sys.stdin.readline().strip())
    A = [int(x) for x in sys.stdin.readline().strip().split()]
    bets.append(A)
X = int(sys.stdin.readline().strip())

# Find the people who bet on X with the fewest bets
min_bets = float('inf')
winners = []
for i in range(N):
    if X in bets[i]:
        if len(bets[i]) < min_bets:
            min_bets = len(bets[i])
            winners = [i+1]
        elif len(bets[i]) == min_bets:
            winners.append(i+1)

# Print the output
print(len(winners))
print(" ".join(map(str, winners)))
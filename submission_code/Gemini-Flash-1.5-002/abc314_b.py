# YOUR CODE HERE
N = int(input())
bets = []
for i in range(N):
    line = list(map(int, input().split()))
    bets.append(line)
X = int(input())

winners = []
min_bets = float('inf')
for i in range(N):
    if X in bets[i][1:]:
        num_bets = bets[i][0]
        if num_bets < min_bets:
            winners = [i + 1]
            min_bets = num_bets
        elif num_bets == min_bets:
            winners.append(i + 1)

print(len(winners))
if len(winners) > 0:
    print(*sorted(winners))
# YOUR CODE HERE
N = int(input())
bets = []
for _ in range(N):
    C = int(input())
    A = list(map(int, input().split()))
    bets.append(A)
X = int(input())

winners = []
min_bets = float('inf')

for i in range(N):
    if X in bets[i]:
        if len(bets[i]) < min_bets:
            winners = [i + 1]
            min_bets = len(bets[i])
        elif len(bets[i]) == min_bets:
            winners.append(i + 1)

print(len(winners))
if winners:
    print(' '.join(map(str, winners)))
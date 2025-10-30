# YOUR CODE HERE
n = int(input())
bets = []
for _ in range(n):
    c = int(input())
    a = list(map(int, input().split()))
    bets.append((c, a))
x = int(input())

winners = []
for i in range(n):
    if x in bets[i][1]:
        winners.append((bets[i][0], i + 1))

if len(winners) == 0:
    print(0)
else:
    min_bets = min(w[0] for w in winners)
    result = [w[1] for w in winners if w[0] == min_bets]
    print(len(result))
    print(*sorted(result))
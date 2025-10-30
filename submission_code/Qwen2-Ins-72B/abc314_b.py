N = int(input())
bets = []
for _ in range(N):
    C = int(input())
    A = list(map(int, input().split()))
    bets.append(A)
X = int(input())

winners = []
min_bets = float('inf')
for i, bet in enumerate(bets):
    if X in bet:
        if len(bet) < min_bets:
            winners = [i+1]
            min_bets = len(bet)
        elif len(bet) == min_bets:
            winners.append(i+1)

print(len(winners))
print(' '.join(map(str, winners)))
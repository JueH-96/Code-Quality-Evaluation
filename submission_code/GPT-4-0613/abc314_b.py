N = int(input().strip())
bets = []
for _ in range(N):
    C = int(input().strip())
    A = list(map(int, input().strip().split()))
    bets.append((C, A))
X = int(input().strip())

winners = [i+1 for i, (C, A) in enumerate(bets) if X in A]
if not winners:
    print(0)
    print()
else:
    min_bets = min(bets[i-1][0] for i in winners)
    winners = [i for i in winners if bets[i-1][0] == min_bets]
    print(len(winners))
    print(' '.join(map(str, winners)))
# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())

bets = {}
for i in range(1, N+1):
    C_i = int(sys.stdin.readline().strip())
    A_i = list(map(int, sys.stdin.readline().strip().split()))
    for a in A_i:
        if a not in bets:
            bets[a] = []
        bets[a].append(i)

X = int(sys.stdin.readline().strip())

if X not in bets:
    print(0)
else:
    min_bets = min(len(bets[X]) for bet in bets.values())
    fewest_betters = [i for i in bets[X] if len(bets[i]) == min_bets]
    print(len(fewest_betters))
    print(' '.join(map(str, sorted(fewest_betters))))
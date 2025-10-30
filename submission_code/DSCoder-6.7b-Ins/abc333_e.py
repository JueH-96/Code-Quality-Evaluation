import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline().strip())
    events = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]

    potions = defaultdict(int)
    max_potions = 0
    actions = []

    for t, x in events:
        if t == 1:
            potions[x] += 1
            max_potions = max(max_potions, potions[x])
            actions.append(1)
        else:
            if potions[x] > 0:
                potions[x] -= 1
                actions.append(1)
            else:
                actions.append(0)

    if len(actions) != N:
        print(-1)
    else:
        print(max_potions)
        print(' '.join(map(str, actions)))

solve()
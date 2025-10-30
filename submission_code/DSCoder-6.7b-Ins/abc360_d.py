import sys
from sortedcontainers import SortedList

def solve():
    N, T = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    X = list(map(int, sys.stdin.readline().split()))

    ants = []
    for i in range(N):
        ants.append((X[i], i, S[i] == '1'))

    ants.sort()

    sl = SortedList()
    pairs = 0
    for x, i, d in ants:
        pairs += len(sl.irange(i))
        if d:
            sl.add(i)
        else:
            sl.remove(i)

    print(pairs)

solve()
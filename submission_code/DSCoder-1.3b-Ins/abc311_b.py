# YOUR CODE HERE

import sys

def solve():
    N, D = map(int, sys.stdin.readline().split())
    schedules = [sys.stdin.readline().strip() for _ in range(N)]

    max_days = 0
    for i in range(D):
        if all(s[i] == 'o' for s in schedules):
            j = i
            while j < D and schedules[0][j] == 'o':
                j += 1
            max_days = max(max_days, j - i)

    print(max_days)

solve()
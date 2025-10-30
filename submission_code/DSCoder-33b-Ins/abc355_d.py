# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        l, r = map(int, input().split())
        intervals.append((l, r))

    intervals.sort()

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if intervals[i][1] >= intervals[j][0]:
                count += 1
            else:
                break

    print(count)

solve()
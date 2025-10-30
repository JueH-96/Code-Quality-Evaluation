# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    pairs = []
    for _ in range(n):
        l, r = map(int, input().split())
        pairs.append((l, r))

    total_sum = sum(l for l, _ in pairs)
    if total_sum > 0 or total_sum + pairs[-1][1] < 0:
        print("No")
        return

    print("Yes")
    x = [pairs[0][1]]
    for i in range(1, n):
        x.append(pairs[i][0])
    x[-1] -= sum(x)
    print(*x)

solve()
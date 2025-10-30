# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        a, b = map(int, input().split())
        giants.append((a, b))

    giants.sort(key=lambda x: x[0] + x[1], reverse=True)

    max_height = 0
    for a, b in giants:
        max_height = max(max_height, a)
        max_height += b

    print(max_height)

solve()
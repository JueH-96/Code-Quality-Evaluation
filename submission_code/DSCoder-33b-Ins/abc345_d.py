# YOUR CODE HERE

import sys

def solve():
    N, H, W = map(int, input().split())
    tiles = [tuple(map(int, input().split())) for _ in range(N)]

    for a in range(1, H+1):
        for b in range(1, W+1):
            if all((a <= x and b <= y) or (a <= y and b <= x) for x, y in tiles):
                print("Yes")
                return
    print("No")

solve()
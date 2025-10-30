# YOUR CODE HERE
import sys

def solve():
    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    B = [list(map(int, input().split())) for _ in range(H)]

    A.sort()
    B.sort()

    if A == B:
        print(0)
    else:
        print(-1)

solve()
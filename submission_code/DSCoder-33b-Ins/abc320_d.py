# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    coords = [None] * (N + 1)
    coords[1] = (0, 0)
    for _ in range(M):
        A, B, X, Y = map(int, sys.stdin.readline().split())
        if coords[A] is None:
            continue
        x, y = coords[A]
        x += X
        y += Y
        if coords[B] is None:
            coords[B] = (x, y)
        elif coords[B] != (x, y):
            print('undecidable')
            return
    for i in range(1, N + 1):
        if coords[i] is None:
            print('undecidable')
        else:
            print(*coords[i])

solve()
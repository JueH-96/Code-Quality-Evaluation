# YOUR CODE HERE

import sys

def solve():
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    items = set()
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        items.add((x, y))

    x, y = 0, 0
    for i in range(N):
        H -= 1
        if H < 0:
            print('No')
            return
        if S[i] == 'R':
            x += 1
        elif S[i] == 'L':
            x -= 1
        elif S[i] == 'U':
            y += 1
        elif S[i] == 'D':
            y -= 1
        if (x, y) in items and H < K:
            H = K
    print('Yes')

solve()
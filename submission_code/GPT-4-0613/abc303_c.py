import sys
from collections import defaultdict

def solve():
    N, M, H, K = map(int, input().split())
    S = input().strip()
    items = defaultdict(int)
    for _ in range(M):
        x, y = map(int, input().split())
        items[(x, y)] = K

    x, y = 0, 0
    for i in range(N):
        if S[i] == 'R':
            x += 1
        elif S[i] == 'L':
            x -= 1
        elif S[i] == 'U':
            y += 1
        elif S[i] == 'D':
            y -= 1
        H -= 1
        if H < 0:
            print('No')
            return
        if (x, y) in items and H < K:
            H = items[(x, y)]
    print('Yes')

solve()
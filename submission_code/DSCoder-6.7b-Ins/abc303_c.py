import sys
from collections import defaultdict

def solve():
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    items = defaultdict(lambda: False)
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        items[(x, y)] = True
    x, y = 0, 0
    for move in S:
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        H -= 1
        if H < 0:
            return 'No'
        if items[(x, y)]:
            H = max(H, K)
    return 'Yes' if H >= 0 else 'No'

print(solve())
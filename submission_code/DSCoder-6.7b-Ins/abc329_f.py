import sys
from collections import defaultdict

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    colors = list(map(int, sys.stdin.readline().split()))
    boxes = defaultdict(set)
    color_count = defaultdict(int)

    for i in range(1, N+1):
        boxes[i].add(colors[i-1])
        color_count[colors[i-1]] += 1

    for _ in range(Q):
        a, b = map(int, sys.stdin.readline().split())
        for color in boxes[a]:
            color_count[color] -= 1
            if color_count[color] == 0:
                del color_count[color]
            boxes[b].add(color)
            color_count[color] += 1
        del boxes[a]
        sys.stdout.write(str(len(color_count)) + '
')

solve()
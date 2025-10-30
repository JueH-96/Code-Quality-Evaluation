import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right, insort_left

H, W, Q = map(int, sys.stdin.readline().split())
walls = defaultdict(list)
for i in range(1, H+1):
    walls[i] = [0, W+1]
for i in range(1, W+1):
    walls[-i] = [0, H+1]
total = H * W
for _ in range(Q):
    x, y = map(int, sys.stdin.readline().split())
    if [y, y] in walls[x]:
        total -= 1
        walls[x].remove([y, y])
        insort_left(walls[-y], x)
    else:
        total -= bisect_right(walls[x], [y, y]) - bisect_left(walls[x], [y, y]) - 1
        total -= bisect_right(walls[-y], x) - bisect_left(walls[-y], x) - 1
        walls[x].remove([walls[x][bisect_left(walls[x], [y, y])-1], y])
        walls[x].remove([y, walls[x][bisect_right(walls[x], [y, y])]])
        walls[-y].remove([walls[-y][bisect_left(walls[-y], x)-1], x])
        walls[-y].remove([x, walls[-y][bisect_right(walls[-y], x)]])
    print(total)
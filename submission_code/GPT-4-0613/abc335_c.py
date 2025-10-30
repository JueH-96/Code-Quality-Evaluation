# YOUR CODE HERE
import sys
from collections import deque

N, Q = map(int, input().split())
parts = deque([(i, 0) for i in range(1, N+1)])
positions = [(i, 0) for i in range(1, N+1)]

for _ in range(Q):
    query = sys.stdin.readline().split()
    if query[0] == '1':
        direction = query[1]
        if direction == 'R':
            x, y = parts[0]
            parts.appendleft((x+1, y))
            parts.pop()
            positions = [(x+1, y) if i == 0 else positions[i-1] for i in range(N)]
        elif direction == 'L':
            x, y = parts[0]
            parts.appendleft((x-1, y))
            parts.pop()
            positions = [(x-1, y) if i == 0 else positions[i-1] for i in range(N)]
        elif direction == 'U':
            x, y = parts[0]
            parts.appendleft((x, y+1))
            parts.pop()
            positions = [(x, y+1) if i == 0 else positions[i-1] for i in range(N)]
        elif direction == 'D':
            x, y = parts[0]
            parts.appendleft((x, y-1))
            parts.pop()
            positions = [(x, y-1) if i == 0 else positions[i-1] for i in range(N)]
    else:
        p = int(query[1])
        print(*positions[p-1])
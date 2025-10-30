import sys
sys.setrecursionlimit(1 << 25)
from collections import deque
input = sys.stdin.readline

N, M = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(N)]

visited = set()
queue = deque()
enqueued = set()

start_x, start_y = 1, 1  # zero-based indexing (since positions start from (2,2))
visited.add((start_x, start_y))
enqueued.add((start_x, start_y))
queue.append((start_x, start_y))

directions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

while queue:
    x, y = queue.popleft()

    for dx, dy in directions:
        nx, ny = x, y

        while True:
            nx += dx
            ny += dy

            if not (0 <= nx < N and 0 <= ny < M):
                break

            if grid[nx][ny] == '#':
                break

            if (nx, ny) not in visited:
                visited.add((nx, ny))

        # The last ice cell before rock is at (nx - dx, ny - dy)
        end_x, end_y = nx - dx, ny - dy

        if (end_x, end_y) not in enqueued:
            enqueued.add((end_x, end_y))
            queue.append((end_x, end_y))

print(len(visited))
import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
grid = [list(sys.stdin.readline().strip()) for _ in range(n)]

touched = set()
visited = set()
start = (1, 1)
touched.add(start)
visited.add(start)
queue = deque([start])

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while queue:
    i, j = queue.popleft()
    for dx, dy in directions:
        path = []
        current_i, current_j = i, j
        while True:
            ni = current_i + dx
            nj = current_j + dy
            if 0 <= ni < n and 0 <= nj < m:
                if grid[ni][nj] == '.':
                    path.append((ni, nj))
                    current_i, current_j = ni, nj
                else:
                    break
            else:
                break
        for square in path:
            touched.add(square)
        if path:
            end = path[-1]
            if end not in visited:
                visited.add(end)
                queue.append(end)

print(len(touched))
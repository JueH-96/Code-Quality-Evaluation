from collections import deque

def bfs(x, y, tx, ty):
    queue = deque([(x, y, 0)])
    visited = set()
    while queue:
        cx, cy, dist = queue.popleft()
        if (cx, cy) == (tx, ty):
            return dist
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx <= 10**8 and 0 <= ny <= 10**8:
                queue.append((nx, ny, dist + 1))
    return 0

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total_dist = 0
for i in range(n-1):
    for j in range(i+1, n):
        x1, y1 = points[i]
        x2, y2 = points[j]
        total_dist += bfs(x1, y1, x2, y2)

print(total_dist)
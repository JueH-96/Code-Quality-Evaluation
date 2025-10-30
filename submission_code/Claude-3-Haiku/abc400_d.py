from collections import deque

def bfs(grid, start, end):
    h, w = len(grid), len(grid[0])
    queue = deque([(start, 0)])
    visited = set([start])

    while queue:
        pos, kicks = queue.popleft()
        if pos == end:
            return kicks

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for i in range(1, 3):
                nx, ny = pos[0] + i * dx, pos[1] + i * dy
                if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == '#':
                    grid[nx][ny] = '.'
                    queue.append(((nx, ny), kicks + 1))
                    visited.add((nx, ny))
                else:
                    break

            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == '.' and (nx, ny) not in visited:
                queue.append(((nx, ny), kicks))
                visited.add((nx, ny))

    return -1

def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    a, b, c, d = map(int, input().split())
    start = (a - 1, b - 1)
    end = (c - 1, d - 1)
    print(bfs(grid, start, end))

solve()
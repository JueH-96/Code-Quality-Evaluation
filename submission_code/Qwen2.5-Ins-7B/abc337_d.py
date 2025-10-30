# YOUR CODE HERE
from collections import deque

H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]

def check_row(row):
    count = 0
    last = -1
    for i, cell in enumerate(row):
        if cell == 'o':
            if last == -1:
                last = i
            else:
                if i - last >= K:
                    return True
                last = i
        elif cell == '.':
            count += 1
    return count >= K

def check_col(col):
    count = 0
    last = -1
    for i, cell in enumerate(col):
        if cell == 'o':
            if last == -1:
                last = i
            else:
                if i - last >= K:
                    return True
                last = i
        elif cell == '.':
            count += 1
    return count >= K

row_results = [check_row(grid[i]) for i in range(H)]
col_results = [check_col(grid[j] for j in range(W)) for i in range(W)]

if any(row_results) or any(col_results):
    print(0)
else:
    queue = deque()
    visited = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                queue.append((i, j, 1))
                visited.add((i, j))
    
    while queue:
        x, y, steps = queue.popleft()
        if steps >= K:
            print(steps - 1)
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] == '.':
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    else:
        print(-1)
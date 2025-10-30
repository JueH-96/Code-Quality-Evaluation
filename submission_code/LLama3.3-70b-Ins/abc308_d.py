from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    target = "snuke"
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W

    queue = deque([(0, 0, 0)])
    visited = {(0, 0, 0)}

    while queue:
        x, y, idx = queue.popleft()
        if (x, y) == (H - 1, W - 1) and idx == 4:
            return "Yes"

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and (nx, ny, (idx + 1) % 5) not in visited:
                if grid[nx][ny] == target[(idx + 1) % 5]:
                    queue.append((nx, ny, (idx + 1) % 5))
                    visited.add((nx, ny, (idx + 1) % 5))

    return "No"

print(solve())
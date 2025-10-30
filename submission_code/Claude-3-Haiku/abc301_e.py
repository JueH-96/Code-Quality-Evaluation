from collections import deque

def bfs(grid, start, goal, T):
    H, W = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], 0, 0)])  # (row, col, moves, candies)
    visited = set()
    max_candies = 0

    while queue:
        row, col, moves, candies = queue.popleft()
        if (row, col) in visited:
            continue
        visited.add((row, col))

        if (row, col) == goal:
            if moves <= T:
                return candies
            else:
                return -1

        if grid[row][col] == 'o':
            candies += 1
            max_candies = max(max_candies, candies)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] != '#':
                queue.append((new_row, new_col, moves + 1, candies))

    return -1

def solve():
    H, W, T = map(int, input().split())
    grid = [list(input()) for _ in range(H)]

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)

    return bfs(grid, start, goal, T)

print(solve())
def is_tak_code(grid, i, j):
    for x in range(i, i+3):
        for y in range(j, j+3):
            if grid[x][y] == '.':
                return False
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        for k in range(1, 3):
            nx, ny = i + dx*k, j + dy*k
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '.':
                return False
    return True

def solve():
    N, M = map(int, input().split())
    grid = [input() for _ in range(N)]
    for i in range(N-8):
        for j in range(M-8):
            if is_tak_code(grid, i, j):
                print(i+1, j+1)

solve()
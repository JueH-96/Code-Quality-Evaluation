def main():
    import sys
    N = int(sys.stdin.read())
    center_row = (N + 1) // 2 - 1  # 0-indexed
    center_col = (N + 1) // 2 - 1  # 0-indexed
    grid = [['' for _ in range(N)] for _ in range(N)]
    visited = [[False for _ in range(N)] for _ in range(N)]
    visited[center_row][center_col] = True

    # Directions: right, down, left, up
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    direction = 0  # start with right

    x, y = 0, 0  # start from top-left corner
    num = 1
    total = N * N - 1

    while num <= total:
        if not visited[x][y]:
            grid[x][y] = str(num)
            visited[x][y] = True
            num += 1

        # Try to move in the current direction
        nx = x + dx[direction]
        ny = y + dy[direction]

        if 0 <= nx < N and 0 <= ny < N and not visited[nx][ny]:
            x, y = nx, ny
        else:
            # Change direction
            direction = (direction + 1) % 4
            x += dx[direction]
            y += dy[direction]

    # Set the center to 'T'
    grid[center_row][center_col] = 'T'

    # Print the grid
    for row in grid:
        print(' '.join(row))

if __name__ == "__main__":
    main()
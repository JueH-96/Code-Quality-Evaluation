import sys

def main():
    data = sys.stdin.readline
    H, W, X, Y = map(int, data().split())
    grid = [data().rstrip('
') for _ in range(H)]
    T = data().strip()

    # convert to 0-based
    x, y = X - 1, Y - 1

    visited = [[False] * W for _ in range(H)]
    count = 0

    for c in T:
        if c == 'U':
            nx, ny = x - 1, y
        elif c == 'D':
            nx, ny = x + 1, y
        elif c == 'L':
            nx, ny = x, y - 1
        else:  # c == 'R'
            nx, ny = x, y + 1

        # move if not a wall
        if grid[nx][ny] != '#':
            x, y = nx, ny
            # if it's a house and not yet counted
            if grid[x][y] == '@' and not visited[x][y]:
                visited[x][y] = True
                count += 1

    # output 1-based position and house count
    print(x + 1, y + 1, count)

if __name__ == "__main__":
    main()
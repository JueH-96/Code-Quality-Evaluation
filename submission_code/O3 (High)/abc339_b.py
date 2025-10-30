import sys

def main() -> None:
    # Read input
    H, W, N = map(int, sys.stdin.readline().split())

    # Grid: False = white ('.'), True = black ('#')
    grid = [[False] * W for _ in range(H)]

    # Directions: 0 = up, 1 = right, 2 = down, 3 = left
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    r, c = 0, 0           # current position (0-based)
    direction = 0         # initially facing up

    for _ in range(N):
        if not grid[r][c]:               # white
            grid[r][c] = True            # paint black
            direction = (direction + 1) % 4   # turn right
        else:                            # black
            grid[r][c] = False           # paint white
            direction = (direction - 1) % 4   # turn left

        # move forward one cell with toroidal wrap-around
        r = (r + dr[direction]) % H
        c = (c + dc[direction]) % W

    # Output the final grid
    out_lines = []
    for row in grid:
        line = ''.join('#' if cell else '.' for cell in row)
        out_lines.append(line)
    sys.stdout.write('
'.join(out_lines))

if __name__ == '__main__':
    main()
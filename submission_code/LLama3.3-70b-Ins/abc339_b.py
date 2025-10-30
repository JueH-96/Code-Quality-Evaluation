import sys

def main():
    H, W, N = map(int, sys.stdin.readline().split())
    grid = [['.' for _ in range(W)] for _ in range(H)]
    x, y = 0, 0
    dx, dy = 0, -1

    for _ in range(N):
        if grid[x][y] == '.':
            grid[x][y] = '#'
            dx, dy = dy, -dx
            x, y = (x + dx) % H, (y + dy) % W
        else:
            grid[x][y] = '.'
            dx, dy = -dy, dx
            x, y = (x + dx) % H, (y + dy) % W

    for row in grid:
        print(''.join(row))

if __name__ == '__main__':
    main()
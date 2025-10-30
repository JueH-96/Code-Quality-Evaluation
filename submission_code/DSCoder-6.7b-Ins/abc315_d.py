import sys

def solve():
    _, _ = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(_)]

    while True:
        marked = [[False]*len(grid[0]) for _ in range(len(grid))]
        removed = False

        # Check rows
        for i in range(len(grid)):
            j = 0
            while j < len(grid[0]) - 1:
                if grid[i][j] == grid[i][j+1] and not marked[i][j] and not marked[i][j+1]:
                    marked[i][j] = marked[i][j+1] = True
                    removed = True
                    j += 2
                else:
                    j += 1

        # Check columns
        for j in range(len(grid[0])):
            i = 0
            while i < len(grid) - 1:
                if grid[i][j] == grid[i+1][j] and not marked[i][j] and not marked[i+1][j]:
                    marked[i][j] = marked[i+1][j] = True
                    removed = True
                    i += 2
                else:
                    i += 1

        if not removed:
            break

        # Remove marked cookies
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if marked[i][j]:
                    grid[i][j] = '.'

    # Count remaining cookies
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != '.':
                count += 1

    print(count)

solve()
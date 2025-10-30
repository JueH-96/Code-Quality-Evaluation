import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    grid = data[2:]

    while True:
        marked = set()

        # Check rows
        for i in range(H):
            row = grid[i]
            if len(set(row)) == 1 and len(row) > 1:
                for j in range(W):
                    marked.add((i, j))

        # Check columns
        for j in range(W):
            col = [grid[i][j] for i in range(H)]
            if len(set(col)) == 1 and len(col) > 1:
                for i in range(H):
                    marked.add((i, j))

        if not marked:
            break

        # Remove marked cookies
        for i, j in marked:
            grid[i] = grid[i][:j] + '.' + grid[i][j+1:]

    # Count remaining cookies
    remaining = sum(1 for row in grid for cell in row if cell != '.')
    print(remaining)

solve()
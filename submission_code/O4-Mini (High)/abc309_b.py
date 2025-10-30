def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    # Read grid as list of list of chars for easy assignment
    grid = [list(input().strip()) for _ in range(N)]

    # Collect coordinates of the outer squares in clockwise order
    border = []
    # Top row (0,0) to (0,N-1)
    for j in range(N):
        border.append((0, j))
    # Right column (1,N-1) to (N-1,N-1)
    for i in range(1, N):
        border.append((i, N-1))
    # Bottom row (N-1,N-2) down to (N-1,0)
    for j in range(N-2, -1, -1):
        border.append((N-1, j))
    # Left column (N-2,0) up to (1,0)
    for i in range(N-2, 0, -1):
        border.append((i, 0))

    # Extract the border values
    vals = [grid[i][j] for i, j in border]
    # Shift clockwise by one: new_vals[k] goes to border[k]
    new_vals = [vals[-1]] + vals[:-1]

    # Write back the shifted values
    for (i, j), v in zip(border, new_vals):
        grid[i][j] = v

    # Print the resulting grid
    out = sys.stdout
    for row in grid:
        out.write(''.join(row) + '
')


if __name__ == '__main__':
    main()
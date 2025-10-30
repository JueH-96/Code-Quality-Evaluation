def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    grid = [list(input().strip()) for _ in range(N)]

    # Collect border coordinates in clockwise order
    coords = []
    # Top row
    for j in range(N):
        coords.append((0, j))
    # Right column (excluding corners)
    for i in range(1, N - 1):
        coords.append((i, N - 1))
    # Bottom row (in reverse)
    for j in range(N - 1, -1, -1):
        coords.append((N - 1, j))
    # Left column (excluding corners, in reverse)
    for i in range(N - 2, 0, -1):
        coords.append((i, 0))

    # Extract the values on the border
    vals = [grid[i][j] for i, j in coords]
    # Shift clockwise by one: last element moves to front
    shifted = [vals[-1]] + vals[:-1]

    # Write back the shifted values
    for (i, j), v in zip(coords, shifted):
        grid[i][j] = v

    # Print the result
    out = sys.stdout
    for row in grid:
        out.write("".join(row) + "
")

if __name__ == "__main__":
    main()
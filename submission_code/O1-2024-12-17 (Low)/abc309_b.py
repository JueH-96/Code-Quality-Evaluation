def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = data[1:]  # Each element is a string representing one row

    # Convert A into a 2D list of integers
    grid = [list(map(int, list(row))) for row in A]

    # Collect the outer ring squares in clockwise order
    # Note: We'll use 0-based indices internally
    outer = []
    # Top row (left to right)
    for j in range(0, N):
        outer.append((0, j))
    # Right column (top to bottom, excluding the corner already taken)
    for i in range(1, N-1):
        outer.append((i, N-1))
    # Bottom row (right to left)
    if N > 1:  # only collect if there's more than 1 row
        for j in range(N-1, -1, -1):
            outer.append((N-1, j))
    # Left column (bottom to top, excluding the corners already taken)
    if N > 1:  # only collect if there's more than 1 column
        for i in range(N-2, 0, -1):
            outer.append((i, 0))

    # Get the values in order
    values = [grid[r][c] for (r, c) in outer]

    # Shift the values by 1 position clockwise, meaning the last goes to first
    shift_values = [values[-1]] + values[:-1]

    # Place them back into the grid
    for (r, c), val in zip(outer, shift_values):
        grid[r][c] = val

    # Print the resulting grid
    for row in grid:
        print("".join(map(str, row)))


# Do not forget to call main()
if __name__ == "__main__":
    main()
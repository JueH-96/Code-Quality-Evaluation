def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    matrix = data[1:]  # N lines of length N
    arr = [list(row) for row in matrix]

    # Collect coordinates of the outer ring in clockwise order.
    path = []
    # top row (left to right)
    for j in range(N):
        path.append((0, j))
    # right column (top to bottom, excluding corners)
    for i in range(1, N - 1):
        path.append((i, N - 1))
    # bottom row (right to left)
    if N > 1:
        for j in range(N - 1, -1, -1):
            path.append((N - 1, j))
    # left column (bottom to top, excluding corners)
    for i in range(N - 2, 0, -1):
        path.append((i, 0))

    # Read the original values from the outer ring in the specified order
    values = [arr[r][c] for (r, c) in path]
    # Shift them clockwise by 1, i.e. rotate to the right
    shifted = [values[-1]] + values[:-1]

    # Place the shifted values back into the array
    for (r, c), val in zip(path, shifted):
        arr[r][c] = val

    # Output the resulting grid
    for i in range(N):
        print("".join(arr[i]))

# Do not forget to call main()
main()
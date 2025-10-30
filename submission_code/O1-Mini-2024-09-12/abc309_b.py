# YOUR CODE HERE
def main():
    import sys

    N = int(sys.stdin.readline())
    grid = [list(sys.stdin.readline().strip()) for _ in range(N)]

    positions = []
    # Top row
    for j in range(N):
        positions.append((0, j))
    # Right column
    for i in range(1, N):
        positions.append((i, N-1))
    # Bottom row
    if N > 1:
        for j in range(N-2, -1, -1):
            positions.append((N-1, j))
    # Left column
    if N > 1:
        for i in range(N-2, 0, -1):
            positions.append((i, 0))

    # Extract the values
    values = [grid[i][j] for i, j in positions]
    # Shift clockwise by one
    values = [values[-1]] + values[:-1]
    # Assign back
    for idx, (i, j) in enumerate(positions):
        grid[i][j] = values[idx]

    # Print the grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()
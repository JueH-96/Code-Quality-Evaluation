import sys

def main() -> None:
    input_data = sys.stdin.read().strip().splitlines()
    if not input_data:
        return
    N = int(input_data[0])
    grid = [list(line.strip()) for line in input_data[1:1 + N]]

    # Coordinates of the outer ring in clockwise order
    coords = []

    # Top row (left → right)
    for j in range(N):
        coords.append((0, j))
    # Right column (top+1 → bottom)
    for i in range(1, N):
        coords.append((i, N - 1))
    # Bottom row (right-1 → left)
    for j in range(N - 2, -1, -1):
        coords.append((N - 1, j))
    # Left column (bottom-1 → top+1)
    for i in range(N - 2, 0, -1):
        coords.append((i, 0))

    # Copy original grid
    new_grid = [row[:] for row in grid]

    # Shift clockwise by one: value at coords[i] moves to coords[i+1]
    total = len(coords)
    for idx, (r, c) in enumerate(coords):
        nr, nc = coords[(idx + 1) % total]
        new_grid[nr][nc] = grid[r][c]

    # Output
    for row in new_grid:
        print("".join(row))

if __name__ == "__main__":
    main()
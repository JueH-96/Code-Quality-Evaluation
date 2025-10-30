def solve():
    n = int(input())
    grid = [input() for _ in range(n)]

    outer_coords = []
    # Top row
    for j in range(n):
        outer_coords.append((0, j))
    # Right column
    for i in range(1, n):
        outer_coords.append((i, n - 1))
    # Bottom row
    for j in range(n - 2, -1, -1):
        outer_coords.append((n - 1, j))
    # Left column
    for i in range(n - 2, 0, -1):
        outer_coords.append((i, 0))

    outer_values = [grid[r][c] for r, c in outer_coords]

    if outer_values:
        shifted_outer_values = [outer_values[-1]] + outer_values[:-1]
    else:
        shifted_outer_values = []

    new_grid = [list(row) for row in grid]

    for i in range(len(outer_coords)):
        r, c = outer_coords[i]
        new_grid[r][c] = shifted_outer_values[i]

    for row in new_grid:
        print("".join(row))

solve()
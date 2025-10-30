import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Read the grid
    grid = []
    for _ in range(N):
        row_str = sys.stdin.readline().strip()
        grid.append([int(c) for c in row_str])

    # Create a new grid for the result. We make a deep copy
    # so that inner elements (not on the outer perimeter) remain unchanged.
    new_grid = [row[:] for row in grid]

    # Step 1: Collect values from the outer squares in clockwise order
    outer_values = []

    # A. Top row: (0,0) to (0, N-1)
    for j in range(N):
        outer_values.append(grid[0][j])

    # B. Right column: (1, N-1) to (N-1, N-1)
    # Start from row 1 because (0, N-1) is already covered by the top row.
    for i in range(1, N):
        outer_values.append(grid[i][N-1])

    # C. Bottom row: (N-1, N-2) down to (N-1, 0)
    # Start from col N-2 because (N-1, N-1) is already covered by the right column.
    for j in range(N - 2, -1, -1):
        outer_values.append(grid[N-1][j])

    # D. Left column: (N-2, 0) down to (1, 0)
    # Start from row N-2 because (N-1, 0) is already covered by the bottom row.
    # End at row 1 because (0, 0) is already covered by the top row.
    for i in range(N - 2, 0, -1):
        outer_values.append(grid[i][0])
    
    # Step 2: Perform the clockwise shift on the collected values
    # The last element moves to the first position, others shift right by one.
    shifted_outer_values = [outer_values[-1]] + outer_values[:-1]

    # Step 3: Place shifted values back into the new_grid
    k = 0 # Index to keep track of current element in shifted_outer_values

    # A. Top row
    for j in range(N):
        new_grid[0][j] = shifted_outer_values[k]
        k += 1

    # B. Right column
    for i in range(1, N):
        new_grid[i][N-1] = shifted_outer_values[k]
        k += 1

    # C. Bottom row
    for j in range(N - 2, -1, -1):
        new_grid[N-1][j] = shifted_outer_values[k]
        k += 1

    # D. Left column
    for i in range(N - 2, 0, -1):
        new_grid[i][0] = shifted_outer_values[k]
        k += 1

    # Step 4: Print the resulting grid
    for r in range(N):
        # Convert integers in the row back to strings and join them without spaces
        print("".join(map(str, new_grid[r])))

solve()
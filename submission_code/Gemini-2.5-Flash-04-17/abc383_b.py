# YOUR CODE HERE
import sys

def solve():
    # Read H, W, D
    h, w, d = map(int, sys.stdin.readline().split())

    # Read the grid
    grid = []
    for _ in range(h):
        grid.append(sys.stdin.readline().strip())

    # Find all floor cell coordinates
    floor_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                floor_cells.append((r, c))

    # Get the number of floor cells
    num_floor_cells = len(floor_cells)

    # Initialize max humidified floor cells count
    max_humidified_floor_cells = 0

    # Iterate through all pairs of distinct floor cells
    # Use indices i and j such that 0 <= i < j < num_floor_cells
    # This ensures we pick two different floor cells as humidifier locations
    for i in range(num_floor_cells):
        for j in range(i + 1, num_floor_cells):
            # Get coordinates of the two humidifiers
            h1_r, h1_c = floor_cells[i]
            h2_r, h2_c = floor_cells[j]

            # Set to store humidified floor cell coordinates for this pair
            # Using a set ensures that each unique floor cell is counted only once
            humidified_floor_cell_coords = set()

            # Iterate through all floor cells to check if they are humidified
            for floor_cell_r, floor_cell_c in floor_cells:
                # Calculate Manhattan distance from the first humidifier
                dist1 = abs(floor_cell_r - h1_r) + abs(floor_cell_c - h1_c)

                # Calculate Manhattan distance from the second humidifier
                dist2 = abs(floor_cell_r - h2_r) + abs(floor_cell_c - h2_c)

                # Check if the floor cell is within distance D of either humidifier
                if dist1 <= d or dist2 <= d:
                    humidified_floor_cell_coords.add((floor_cell_r, floor_cell_c))

            # The number of humidified floor cells for this pair is the size of the set
            current_humidified_count = len(humidified_floor_cell_coords)

            # Update the maximum count found so far
            max_humidified_floor_cells = max(max_humidified_floor_cells, current_humidified_count)

    # Print the result
    print(max_humidified_floor_cells)

solve()
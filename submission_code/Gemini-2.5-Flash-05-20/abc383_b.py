import sys

def solve():
    # Read H, W, D from the first line of standard input
    H, W, D = map(int, sys.stdin.readline().split())

    # Read the grid S. Each row is read as a string.
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Step 1: Identify all floor cells ('.') and store their coordinates.
    # Coordinates are stored as (row, column) tuples.
    floor_cells = []
    for r in range(H):
        for c in range(W):
            if S[r][c] == '.':
                floor_cells.append((r, c))

    # Initialize the maximum number of humidified floor cells found.
    # Since there are at least two floor cells guaranteed, and we place
    # two humidifiers on them, the minimum possible humidified count is 2.
    max_humidified_floor_cells = 0

    # Get the total number of floor cells found.
    N_floor = len(floor_cells)

    # Step 2: Iterate through all unique pairs of floor cells to choose as humidifier locations.
    # We use nested loops to pick two distinct floor cells.
    # h1_idx is the index of the first humidifier's coordinates in the floor_cells list.
    for h1_idx in range(N_floor):
        # h2_idx is the index of the second humidifier's coordinates.
        # It starts from h1_idx + 1 to ensure:
        # 1. The two humidifiers are distinct (h1_idx != h2_idx).
        # 2. Each pair is considered only once (e.g., (A, B) is processed, not (B, A)).
        for h2_idx in range(h1_idx + 1, N_floor):
            # Get the actual coordinates for the two chosen humidifier locations.
            h1_coords = floor_cells[h1_idx]
            h2_coords = floor_cells[h2_idx]

            # Step 3: For the current pair of humidifiers (h1_coords, h2_coords),
            # determine which floor cells are humidified.
            # A set is used to store humidified floor cells to automatically handle uniqueness.
            current_humidified_set = set()

            # Iterate through every floor cell in the grid.
            for fr, fc in floor_cells:
                # Calculate the Manhattan distance from the current floor cell (fr, fc)
                # to the first humidifier h1_coords.
                dist_to_h1 = abs(fr - h1_coords[0]) + abs(fc - h1_coords[1])
                
                # Calculate the Manhattan distance from the current floor cell (fr, fc)
                # to the second humidifier h2_coords.
                dist_to_h2 = abs(fr - h2_coords[0]) + abs(fc - h2_coords[1])

                # A floor cell is considered humidified if its Manhattan distance
                # to AT LEAST ONE of the humidifiers is less than or equal to D.
                if dist_to_h1 <= D or dist_to_h2 <= D:
                    current_humidified_set.add((fr, fc))
            
            # Step 4: Update the maximum count.
            # After checking all floor cells for the current pair of humidifiers,
            # compare the number of humidified floor cells found with the overall maximum.
            max_humidified_floor_cells = max(max_humidified_floor_cells, len(current_humidified_set))

    # Print the final maximum number of humidified floor cells.
    print(max_humidified_floor_cells)

# This ensures that the solve() function is called only when the script is executed directly.
if __name__ == '__main__':
    solve()
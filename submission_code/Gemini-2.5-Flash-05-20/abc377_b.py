# Read the 8 strings representing the grid
grid = []
for _ in range(8):
    grid.append(input())

# Identify all rows that contain at least one existing piece
# And all columns that contain at least one existing piece
occupied_rows = set()
occupied_cols = set()

# Iterate through each square in the 8x8 grid (0-indexed)
for r in range(8): # Row index from 0 to 7
    for c in range(8): # Column index from 0 to 7
        if grid[r][c] == '#':
            # If an existing piece is found, mark its row and column as occupied
            occupied_rows.add(r)
            occupied_cols.add(c)

# Initialize a counter for safe squares
safe_squares_count = 0

# Iterate through all 64 squares to check if they are safe placement spots
for r in range(8):
    for c in range(8):
        # A square (r, c) is a safe spot to place a new piece if:
        # 1. It is currently empty (represented by '.')
        # 2. Its row 'r' does not contain any existing piece (meaning it's not in occupied_rows)
        # 3. Its column 'c' does not contain any existing piece (meaning it's not in occupied_cols)
        
        if grid[r][c] == '.':
            if r not in occupied_rows and c not in occupied_cols:
                safe_squares_count += 1

# Print the total number of safe squares found
print(safe_squares_count)
# Read N, the number of rectangular sheets
N = int(input())

# Initialize a 100x100 grid. grid[x_coord][y_coord] will be 1 if the
# unit square with bottom-left corner (x_coord, y_coord) is covered, 0 otherwise.
# x_coord and y_coord will range from 0 to 99.
grid = [[0 for _ in range(100)] for _ in range(100)]

# Process each rectangular sheet
for _ in range(N):
    A, B, C, D = map(int, input().split())
    
    # Mark all unit squares covered by this sheet.
    # A sheet covering A <= x <= B, C <= y <= D covers unit squares (x_coord, y_coord)
    # where A <= x_coord < B and C <= y_coord < D.
    # In Python, range(L, U) generates integers from L up to U-1.
    for x_coord in range(A, B):
        for y_coord in range(C, D):
            # The problem constraints (0 <= A_i < B_i <= 100, etc.) ensure
            # that A, B, C, D are such that x_coord and y_coord will be
            # valid indices (0-99) for the 100x100 grid.
            grid[x_coord][y_coord] = 1

# Calculate the total area covered
total_area = 0
# Iterate over all possible unit squares in the 100x100 area
for x_coord in range(100):
    for y_coord in range(100):
        # Each marked square (value 1) contributes 1 to the total area.
        # Unmarked squares (value 0) contribute 0.
        total_area += grid[x_coord][y_coord]
            
# Print the total area S
print(total_area)
def calculate_area_of_sheets(N, sheets):
    # Create a grid to mark covered areas
    grid_size = 101  # Since coordinates range from 0 to 100
    grid = [[0] * grid_size for _ in range(grid_size)]
    
    # Mark the grid based on the sheets
    for A, B, C, D in sheets:
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = 1  # Mark the cell as covered
    
    # Calculate the total area covered
    area = sum(sum(row) for row in grid)
    return area

import sys

# Read input
input = sys.stdin.read
data = input().strip().splitlines()
N = int(data[0])
sheets = [tuple(map(int, line.split())) for line in data[1:N+1]]

# Calculate and print the area
result = calculate_area_of_sheets(N, sheets)
print(result)
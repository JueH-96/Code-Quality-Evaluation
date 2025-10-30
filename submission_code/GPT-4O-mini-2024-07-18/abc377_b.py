def count_safe_squares(grid):
    # Create a set to track the rows and columns that are attacked
    attacked_rows = set()
    attacked_columns = set()
    
    # Iterate through the grid to find existing pieces
    for i in range(8):
        for j in range(8):
            if grid[i][j] == '#':
                attacked_rows.add(i)
                attacked_columns.add(j)
    
    # Count the number of safe squares
    safe_count = 0
    for i in range(8):
        for j in range(8):
            if grid[i][j] == '.' and i not in attacked_rows and j not in attacked_columns:
                safe_count += 1
    
    return safe_count

# Read input
import sys
input = sys.stdin.read
data = input().strip().splitlines()

# Call the function and print the result
result = count_safe_squares(data)
print(result)
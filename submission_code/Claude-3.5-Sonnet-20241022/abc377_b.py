# Read the 8x8 grid
grid = []
for _ in range(8):
    row = input().strip()
    grid.append(row)

def is_safe(row, col, grid):
    # Check if the square is empty
    if grid[row][col] == '#':
        return False
    
    # Check if there's any piece in the same row
    if '#' in grid[row]:
        return False
        
    # Check if there's any piece in the same column
    for i in range(8):
        if grid[i][col] == '#':
            return False
            
    return True

# Count safe squares
safe_count = 0
for i in range(8):
    for j in range(8):
        if is_safe(i, j, grid):
            safe_count += 1

print(safe_count)
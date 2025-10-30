def can_form_rectangle(H, W, grid):
    # Find all black cells
    black_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                black_cells.append((i, j))
    
    # Get boundaries of existing black cells
    min_row = min(i for i, _ in black_cells)
    max_row = max(i for i, _ in black_cells)
    min_col = min(j for _, j in black_cells)
    max_col = max(j for _, j in black_cells)
    
    # Check if all cells within these boundaries can be black
    # and all cells outside must be white
    for i in range(H):
        for j in range(W):
            cell = grid[i][j]
            is_in_rectangle = (min_row <= i <= max_row and min_col <= j <= max_col)
            
            # If cell is already painted
            if cell != '?':
                # If black cell outside rectangle or white cell inside rectangle
                if (cell == '#' and not is_in_rectangle) or (cell == '.' and is_in_rectangle):
                    return False
    
    return True

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

# Print result
print("Yes" if can_form_rectangle(H, W, grid) else "No")
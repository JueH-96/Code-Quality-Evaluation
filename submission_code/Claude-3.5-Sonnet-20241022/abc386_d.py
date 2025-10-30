def check_possible(N, M, colored_cells):
    # For each row and column, we need to track:
    # 1. The leftmost/topmost white cell (which forces all cells after it to be white)
    # 2. The rightmost/bottommost black cell (which forces all cells before it to be black)
    
    row_first_white = {}  # minimum position where white appears in row
    row_last_black = {}   # maximum position where black appears in row
    col_first_white = {}  # minimum position where white appears in column
    col_last_black = {}   # maximum position where black appears in column
    
    # Process all colored cells
    for x, y, c in colored_cells:
        if c == 'W':
            # Update row constraints
            if x not in row_first_white:
                row_first_white[x] = y
            else:
                row_first_white[x] = min(row_first_white[x], y)
            
            # Update column constraints
            if y not in col_first_white:
                col_first_white[y] = x
            else:
                col_first_white[y] = min(col_first_white[y], x)
        else:  # c == 'B'
            # Update row constraints
            if x not in row_last_black:
                row_last_black[x] = y
            else:
                row_last_black[x] = max(row_last_black[x], y)
            
            # Update column constraints
            if y not in col_last_black:
                col_last_black[y] = x
            else:
                col_last_black[y] = max(col_last_black[y], x)
    
    # Check if constraints are violated
    
    # Check rows
    for row in range(1, N+1):
        first_white = row_first_white.get(row, N+1)
        last_black = row_last_black.get(row, 0)
        if last_black >= first_white:
            return False
            
    # Check columns
    for col in range(1, N+1):
        first_white = col_first_white.get(col, N+1)
        last_black = col_last_black.get(col, 0)
        if last_black >= first_white:
            return False
    
    return True

# Read input
N, M = map(int, input().split())
colored_cells = []
for _ in range(M):
    x, y, c = input().split()
    colored_cells.append((int(x), int(y), c))

# Print result
print("Yes" if check_possible(N, M, colored_cells) else "No")
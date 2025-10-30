# YOUR CODE HERE
n, m = map(int, input().split())
grid = []
for _ in range(n):
    grid.append(input().strip())

# Check all possible 9x9 regions
for i in range(n - 8):  # Top-left row can be from 0 to n-9
    for j in range(m - 8):  # Top-left column can be from 0 to m-9
        # Check if this 9x9 region is a TaK Code
        is_tak_code = True
        
        # Check top-left 3x3 region (all must be black)
        for di in range(3):
            for dj in range(3):
                if grid[i + di][j + dj] != '#':
                    is_tak_code = False
                    break
            if not is_tak_code:
                break
        
        if not is_tak_code:
            continue
        
        # Check bottom-right 3x3 region (all must be black)
        for di in range(6, 9):
            for dj in range(6, 9):
                if grid[i + di][j + dj] != '#':
                    is_tak_code = False
                    break
            if not is_tak_code:
                break
        
        if not is_tak_code:
            continue
        
        # Check cells adjacent to top-left 3x3 region (must be white)
        # Right edge of top-left 3x3
        for di in range(3):
            if grid[i + di][j + 3] != '.':
                is_tak_code = False
                break
        
        if not is_tak_code:
            continue
        
        # Bottom edge of top-left 3x3 (including diagonal)
        for dj in range(4):
            if grid[i + 3][j + dj] != '.':
                is_tak_code = False
                break
        
        if not is_tak_code:
            continue
        
        # Check cells adjacent to bottom-right 3x3 region (must be white)
        # Left edge of bottom-right 3x3
        for di in range(6, 9):
            if grid[i + di][j + 5] != '.':
                is_tak_code = False
                break
        
        if not is_tak_code:
            continue
        
        # Top edge of bottom-right 3x3 (including diagonal)
        for dj in range(5, 9):
            if grid[i + 5][j + dj] != '.':
                is_tak_code = False
                break
        
        if is_tak_code:
            print(i + 1, j + 1)  # Convert to 1-indexed
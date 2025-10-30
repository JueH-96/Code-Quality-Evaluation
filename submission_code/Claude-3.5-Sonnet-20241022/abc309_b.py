N = int(input())
grid = []
for _ in range(N):
    row = list(map(int, list(input())))
    grid.append(row)

# If N is 2 or less, no change needed
if N <= 2:
    for row in grid:
        print(''.join(map(str, row)))
else:
    # Store the outer values in clockwise order
    outer = []
    
    # Top row
    for j in range(N):
        outer.append(grid[0][j])
        
    # Right column (except top)
    for i in range(1, N):
        outer.append(grid[i][N-1])
        
    # Bottom row (except rightmost)
    for j in range(N-2, -1, -1):
        outer.append(grid[N-1][j])
        
    # Left column (except bottom and top)
    for i in range(N-2, 0, -1):
        outer.append(grid[i][0])
    
    # Rotate array by 1 position
    last = outer[-1]
    outer = [last] + outer[:-1]
    
    # Put values back into grid
    idx = 0
    
    # Top row
    for j in range(N):
        grid[0][j] = outer[idx]
        idx += 1
        
    # Right column (except top)
    for i in range(1, N):
        grid[i][N-1] = outer[idx]
        idx += 1
        
    # Bottom row (except rightmost)
    for j in range(N-2, -1, -1):
        grid[N-1][j] = outer[idx]
        idx += 1
        
    # Left column (except bottom and top)
    for i in range(N-2, 0, -1):
        grid[i][0] = outer[idx]
        idx += 1
    
    # Print result
    for row in grid:
        print(''.join(map(str, row)))
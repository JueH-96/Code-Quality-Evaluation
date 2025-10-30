import sys

def solve():
    N, M = map(int, input().split())
    
    # Create a 2D grid to store the colors
    grid = [['W' for _ in range(N)] for _ in range(N)]
    
    # Mark the cells that are already colored
    for _ in range(M):
        x, y, c = map(str, input().split())
        x, y = int(x), int(y)
        grid[x-1][y-1] = c
    
    # Check if the remaining cells can be colored to satisfy the conditions
    for row in range(N):
        for col in range(N):
            if grid[row][col] == 'W':
                # Find the leftmost black cell in the row
                left_black = -1
                for i in range(col):
                    if grid[row][i] == 'B':
                        left_black = i
                
                # Find the topmost black cell in the column
                top_black = -1
                for i in range(row):
                    if grid[i][col] == 'B':
                        top_black = i
                
                # If the leftmost black cell in the row is not to the left of the current cell
                # or the topmost black cell in the column is not above the current cell,
                # then the conditions cannot be satisfied
                if left_black != -1 and left_black >= col:
                    return "No"
                if top_black != -1 and top_black >= row:
                    return "No"
                
                # Color the current cell black
                grid[row][col] = 'B'
    
    return "Yes"

print(solve())
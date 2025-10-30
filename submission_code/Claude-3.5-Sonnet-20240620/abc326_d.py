# YOUR CODE HERE
def solve():
    N = int(input())
    R = input().strip()
    C = input().strip()

    def dfs(grid, row, col):
        if row == N:
            return True
        
        if col == N:
            return dfs(grid, row + 1, 0)
        
        if grid[row][col] != '.':
            return dfs(grid, row, col + 1)
        
        for char in 'ABC':
            if (char not in grid[row] and 
                char not in [grid[i][col] for i in range(N)] and
                (col == 0 and char == R[row] or col != 0) and
                (row == 0 and char == C[col] or row != 0)):
                
                grid[row][col] = char
                if dfs(grid, row, col + 1):
                    return True
                grid[row][col] = '.'
        
        return False

    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    if dfs(grid, 0, 0):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")

solve()
def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    visited = set()
    
    def move(row, col, direction):
        if direction == "up":
            while row > 0 and grid[row-1][col] == '.':
                row -= 1
            return row, col
        elif direction == "down":
            while row < n - 1 and grid[row+1][col] == '.':
                row += 1
            return row, col
        elif direction == "left":
            while col > 0 and grid[row][col-1] == '.':
                col -= 1
            return row, col
        elif direction == "right":
            while col < m - 1 and grid[row][col+1] == '.':
                col += 1
            return row, col
        
    def dfs(row, col):
        if (row, col) in visited:
            return
        
        visited.add((row, col))
        
        new_row, new_col = move(row, col, "up")
        dfs(new_row, new_col)
        
        new_row, new_col = move(row, col, "down")
        dfs(new_row, new_col)
        
        new_row, new_col = move(row, col, "left")
        dfs(new_row, new_col)
        
        new_row, new_col = move(row, col, "right")
        dfs(new_row, new_col)
    
    dfs(1, 1)
    print(len(visited))

solve()
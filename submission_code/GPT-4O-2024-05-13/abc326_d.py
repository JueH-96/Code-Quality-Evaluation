# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    R = data[1]
    C = data[2]
    
    def is_valid(grid):
        for i in range(N):
            row_set = set(grid[i])
            col_set = set(grid[j][i] for j in range(N))
            if len(row_set) != N or len(col_set) != N:
                return False
        return True
    
    def backtrack(grid, row, col):
        if row == N:
            if is_valid(grid):
                return grid
            else:
                return None
        
        next_row = row + (col + 1) // N
        next_col = (col + 1) % N
        
        for char in 'ABC':
            grid[row][col] = char
            if col == 0 and grid[row][0] != R[row]:
                continue
            if row == 0 and grid[0][col] != C[col]:
                continue
            result = backtrack(grid, next_row, next_col)
            if result:
                return result
            grid[row][col] = '.'
        
        return None
    
    grid = [['.' for _ in range(N)] for _ in range(N)]
    result = backtrack(grid, 0, 0)
    
    if result:
        print("Yes")
        for row in result:
            print("".join(row))
    else:
        print("No")

# Uncomment the following line to run the function
# solve()
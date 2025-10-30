def check_valid(grid, n):
    # Check rows
    for row in grid:
        chars = [c for c in row if c in 'ABC']
        if len(chars) != 3 or len(set(chars)) != 3:
            return False
            
    # Check columns
    for j in range(n):
        chars = [grid[i][j] for i in range(n) if grid[i][j] in 'ABC']
        if len(chars) != 3 or len(set(chars)) != 3:
            return False
            
    return True

def get_leftmost(row):
    for c in row:
        if c in 'ABC':
            return c
    return None

def get_topmost(grid, col, n):
    for i in range(n):
        if grid[i][col] in 'ABC':
            return grid[i][col]
    return None

def solve():
    n = int(input())
    r = input()
    c = input()
    
    # Try all possible grids
    for mask in range(3**(n*n)):
        grid = [['.' for _ in range(n)] for _ in range(n)]
        tmp = mask
        
        # Generate grid from mask
        for i in range(n):
            for j in range(n):
                val = tmp % 3
                tmp //= 3
                if val == 0:
                    grid[i][j] = 'A'
                elif val == 1:
                    grid[i][j] = 'B'
                elif val == 2:
                    grid[i][j] = 'C'
                    
        # Check if valid
        if not check_valid(grid, n):
            continue
            
        # Check leftmost condition
        valid = True
        for i in range(n):
            left = get_leftmost(grid[i])
            if left != r[i]:
                valid = False
                break
                
        if not valid:
            continue
            
        # Check topmost condition
        for j in range(n):
            top = get_topmost(grid, j, n)
            if top != c[j]:
                valid = False
                break
                
        if valid:
            print("Yes")
            for row in grid:
                print("".join(row))
            return
            
    print("No")

solve()
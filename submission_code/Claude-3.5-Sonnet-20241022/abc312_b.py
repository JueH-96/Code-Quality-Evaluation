def is_tak_code(grid, top, left):
    # Check top-left 3x3 region is black
    for i in range(3):
        for j in range(3):
            if grid[top+i][left+j] != '#':
                return False
                
    # Check bottom-right 3x3 region is black        
    for i in range(6,9):
        for j in range(6,9):
            if grid[top+i][left+j] != '#':
                return False
    
    # Check cells adjacent to top-left region are white
    adjacent = [
        (0,3), (1,3), (2,3), (3,0), (3,1), (3,2), (3,3),
        (0,4), (1,4), (2,4), (4,0), (4,1), (4,2)
    ]
    for i,j in adjacent:
        if grid[top+i][left+j] != '.':
            return False
            
    # Check cells adjacent to bottom-right region are white
    adjacent = [
        (5,6), (5,7), (5,8), (6,5), (7,5), (8,5),
        (4,6), (4,7), (4,8), (6,4), (7,4), (8,4)
    ]
    for i,j in adjacent:
        if grid[top+i][left+j] != '.':
            return False
            
    return True

N, M = map(int, input().split())
grid = []
for _ in range(N):
    grid.append(input())

# Try all possible 9x9 regions
for i in range(N-8):
    for j in range(M-8):
        if is_tak_code(grid, i, j):
            print(i+1, j+1)